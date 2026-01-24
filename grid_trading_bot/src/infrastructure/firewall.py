"""
Firewall Rules Management Module.

Provides IP whitelist/blacklist management, port access control,
and request filtering for application-level security.
"""

import asyncio
import ipaddress
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.core import get_logger

logger = get_logger(__name__)


class RuleAction(Enum):
    """Firewall rule action."""

    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"  # Allow but log


class RuleType(Enum):
    """Firewall rule type."""

    IP = "ip"
    CIDR = "cidr"
    PORT = "port"
    PATH = "path"
    RATE = "rate"


class RulePriority(Enum):
    """Rule priority (higher number = higher priority)."""

    LOW = 10
    MEDIUM = 50
    HIGH = 90
    CRITICAL = 100


@dataclass
class FirewallRule:
    """Firewall rule definition."""

    rule_id: str
    rule_type: RuleType
    action: RuleAction
    value: str  # IP, CIDR, port, path pattern
    priority: RulePriority = RulePriority.MEDIUM
    description: str = ""
    enabled: bool = True
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    last_hit: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "action": self.action.value,
            "value": self.value,
            "priority": self.priority.value,
            "description": self.description,
            "enabled": self.enabled,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "hit_count": self.hit_count,
            "last_hit": self.last_hit.isoformat() if self.last_hit else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirewallRule":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            rule_type=RuleType(data["rule_type"]),
            action=RuleAction(data["action"]),
            value=data["value"],
            priority=RulePriority(data.get("priority", RulePriority.MEDIUM.value)),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            hit_count=data.get("hit_count", 0),
            last_hit=(
                datetime.fromisoformat(data["last_hit"])
                if data.get("last_hit")
                else None
            ),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "system"),
            tags=data.get("tags", {}),
        )

    def is_expired(self) -> bool:
        """Check if rule has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class RequestContext:
    """Context for request filtering."""

    ip_address: str
    port: int = 0
    path: str = ""
    method: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FilterResult:
    """Result of firewall filtering."""

    allowed: bool
    matched_rule: Optional[FirewallRule] = None
    reason: str = ""
    action_taken: RuleAction = RuleAction.ALLOW


@dataclass
class FirewallConfig:
    """Firewall configuration."""

    # Default action when no rules match
    default_action: RuleAction = RuleAction.DENY

    # Paths
    rules_file: Path = Path("config/firewall_rules.json")

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 100

    # Auto-ban
    enable_auto_ban: bool = True
    auto_ban_threshold: int = 10  # violations before ban
    auto_ban_duration_minutes: int = 60

    # Logging
    log_allowed: bool = False
    log_denied: bool = True

    # Trusted proxies (for X-Forwarded-For)
    trusted_proxies: List[str] = field(default_factory=list)


class FirewallManager:
    """
    Application-level firewall manager.

    Provides IP filtering, port access control, path-based
    filtering, and rate limiting.
    """

    def __init__(self, config: Optional[FirewallConfig] = None):
        """
        Initialize firewall manager.

        Args:
            config: Firewall configuration
        """
        self._config = config or FirewallConfig()
        self._rules: Dict[str, FirewallRule] = {}
        self._rule_counter = 0

        # IP sets for fast lookup
        self._ip_whitelist: Set[str] = set()
        self._ip_blacklist: Set[str] = set()
        self._cidr_rules: List[Tuple[ipaddress.IPv4Network, RuleAction]] = []

        # Rate limiting state
        self._request_counts: Dict[str, List[datetime]] = {}
        self._violations: Dict[str, int] = {}
        self._auto_bans: Dict[str, datetime] = {}

        # Callbacks
        self._on_violation: List[Callable[[str, str], Any]] = []

        # Load rules from file
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from file."""
        if not self._config.rules_file.exists():
            logger.debug("No firewall rules file found")
            return

        try:
            with open(self._config.rules_file, "r") as f:
                data = json.load(f)

            for rule_data in data.get("rules", []):
                rule = FirewallRule.from_dict(rule_data)
                self._rules[rule.rule_id] = rule
                self._update_caches(rule)

            logger.info(f"Loaded {len(self._rules)} firewall rules")

        except Exception as e:
            logger.error(f"Failed to load firewall rules: {e}")

    def _save_rules(self) -> None:
        """Save rules to file."""
        try:
            self._config.rules_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self._config.rules_file, "w") as f:
                json.dump(
                    {"rules": [r.to_dict() for r in self._rules.values()]},
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Failed to save firewall rules: {e}")

    def _update_caches(self, rule: FirewallRule) -> None:
        """Update IP caches for fast lookup."""
        if not rule.enabled or rule.is_expired():
            return

        if rule.rule_type == RuleType.IP:
            if rule.action == RuleAction.ALLOW:
                self._ip_whitelist.add(rule.value)
            elif rule.action == RuleAction.DENY:
                self._ip_blacklist.add(rule.value)

        elif rule.rule_type == RuleType.CIDR:
            try:
                network = ipaddress.ip_network(rule.value, strict=False)
                self._cidr_rules.append((network, rule.action))
                # Sort by specificity (more specific first)
                self._cidr_rules.sort(key=lambda x: x[0].prefixlen, reverse=True)
            except ValueError as e:
                logger.error(f"Invalid CIDR: {rule.value}: {e}")

    def _rebuild_caches(self) -> None:
        """Rebuild all caches from rules."""
        self._ip_whitelist.clear()
        self._ip_blacklist.clear()
        self._cidr_rules.clear()

        for rule in self._rules.values():
            self._update_caches(rule)

    def add_rule(
        self,
        rule_type: RuleType,
        action: RuleAction,
        value: str,
        priority: RulePriority = RulePriority.MEDIUM,
        description: str = "",
        expires_at: Optional[datetime] = None,
        created_by: str = "system",
        tags: Optional[Dict[str, str]] = None,
    ) -> FirewallRule:
        """
        Add a firewall rule.

        Args:
            rule_type: Type of rule
            action: Action to take
            value: Rule value (IP, CIDR, port, path)
            priority: Rule priority
            description: Rule description
            expires_at: Expiration time
            created_by: Creator
            tags: Optional tags

        Returns:
            Created rule
        """
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        rule = FirewallRule(
            rule_id=rule_id,
            rule_type=rule_type,
            action=action,
            value=value,
            priority=priority,
            description=description,
            expires_at=expires_at,
            created_by=created_by,
            tags=tags or {},
        )

        self._rules[rule_id] = rule
        self._update_caches(rule)
        self._save_rules()

        logger.info(f"Added firewall rule: {rule_id} ({action.value} {value})")
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a firewall rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            True if removed
        """
        if rule_id not in self._rules:
            return False

        del self._rules[rule_id]
        self._rebuild_caches()
        self._save_rules()

        logger.info(f"Removed firewall rule: {rule_id}")
        return True

    def update_rule(
        self,
        rule_id: str,
        enabled: Optional[bool] = None,
        action: Optional[RuleAction] = None,
        priority: Optional[RulePriority] = None,
        description: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[FirewallRule]:
        """
        Update a firewall rule.

        Args:
            rule_id: Rule ID to update
            enabled: New enabled state
            action: New action
            priority: New priority
            description: New description
            expires_at: New expiration

        Returns:
            Updated rule or None
        """
        if rule_id not in self._rules:
            return None

        rule = self._rules[rule_id]

        if enabled is not None:
            rule.enabled = enabled
        if action is not None:
            rule.action = action
        if priority is not None:
            rule.priority = priority
        if description is not None:
            rule.description = description
        if expires_at is not None:
            rule.expires_at = expires_at

        self._rebuild_caches()
        self._save_rules()

        return rule

    def get_rule(self, rule_id: str) -> Optional[FirewallRule]:
        """Get a specific rule."""
        return self._rules.get(rule_id)

    def list_rules(
        self,
        rule_type: Optional[RuleType] = None,
        action: Optional[RuleAction] = None,
        enabled_only: bool = False,
    ) -> List[FirewallRule]:
        """
        List firewall rules.

        Args:
            rule_type: Filter by type
            action: Filter by action
            enabled_only: Only return enabled rules

        Returns:
            List of rules
        """
        rules = list(self._rules.values())

        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        if action:
            rules = [r for r in rules if r.action == action]
        if enabled_only:
            rules = [r for r in rules if r.enabled and not r.is_expired()]

        # Sort by priority
        rules.sort(key=lambda r: r.priority.value, reverse=True)
        return rules

    # === Whitelist/Blacklist Convenience Methods ===

    def whitelist_ip(
        self,
        ip: str,
        description: str = "",
        expires_at: Optional[datetime] = None,
    ) -> FirewallRule:
        """Add IP to whitelist."""
        return self.add_rule(
            rule_type=RuleType.IP,
            action=RuleAction.ALLOW,
            value=ip,
            priority=RulePriority.HIGH,
            description=description or f"Whitelisted IP: {ip}",
            expires_at=expires_at,
        )

    def blacklist_ip(
        self,
        ip: str,
        description: str = "",
        expires_at: Optional[datetime] = None,
    ) -> FirewallRule:
        """Add IP to blacklist."""
        return self.add_rule(
            rule_type=RuleType.IP,
            action=RuleAction.DENY,
            value=ip,
            priority=RulePriority.HIGH,
            description=description or f"Blacklisted IP: {ip}",
            expires_at=expires_at,
        )

    def whitelist_cidr(
        self,
        cidr: str,
        description: str = "",
    ) -> FirewallRule:
        """Add CIDR range to whitelist."""
        return self.add_rule(
            rule_type=RuleType.CIDR,
            action=RuleAction.ALLOW,
            value=cidr,
            priority=RulePriority.MEDIUM,
            description=description or f"Whitelisted CIDR: {cidr}",
        )

    def blacklist_cidr(
        self,
        cidr: str,
        description: str = "",
    ) -> FirewallRule:
        """Add CIDR range to blacklist."""
        return self.add_rule(
            rule_type=RuleType.CIDR,
            action=RuleAction.DENY,
            value=cidr,
            priority=RulePriority.MEDIUM,
            description=description or f"Blacklisted CIDR: {cidr}",
        )

    def allow_port(
        self,
        port: int,
        description: str = "",
    ) -> FirewallRule:
        """Allow access to port."""
        return self.add_rule(
            rule_type=RuleType.PORT,
            action=RuleAction.ALLOW,
            value=str(port),
            priority=RulePriority.MEDIUM,
            description=description or f"Allowed port: {port}",
        )

    def deny_port(
        self,
        port: int,
        description: str = "",
    ) -> FirewallRule:
        """Deny access to port."""
        return self.add_rule(
            rule_type=RuleType.PORT,
            action=RuleAction.DENY,
            value=str(port),
            priority=RulePriority.MEDIUM,
            description=description or f"Denied port: {port}",
        )

    def allow_path(
        self,
        path_pattern: str,
        description: str = "",
    ) -> FirewallRule:
        """Allow access to path pattern (regex)."""
        return self.add_rule(
            rule_type=RuleType.PATH,
            action=RuleAction.ALLOW,
            value=path_pattern,
            priority=RulePriority.MEDIUM,
            description=description or f"Allowed path: {path_pattern}",
        )

    def deny_path(
        self,
        path_pattern: str,
        description: str = "",
    ) -> FirewallRule:
        """Deny access to path pattern (regex)."""
        return self.add_rule(
            rule_type=RuleType.PATH,
            action=RuleAction.DENY,
            value=path_pattern,
            priority=RulePriority.MEDIUM,
            description=description or f"Denied path: {path_pattern}",
        )

    # === Request Filtering ===

    def check_request(self, context: RequestContext) -> FilterResult:
        """
        Check if request should be allowed.

        Args:
            context: Request context

        Returns:
            Filter result
        """
        ip = context.ip_address

        # Check auto-ban first
        if ip in self._auto_bans:
            ban_expires = self._auto_bans[ip]
            if datetime.now(timezone.utc) < ban_expires:
                return FilterResult(
                    allowed=False,
                    reason="Auto-banned due to violations",
                    action_taken=RuleAction.DENY,
                )
            else:
                # Ban expired
                del self._auto_bans[ip]
                self._violations.pop(ip, None)

        # Check rate limiting
        if self._config.enable_rate_limiting:
            if self._is_rate_limited(ip):
                self._record_violation(ip, "Rate limit exceeded")
                return FilterResult(
                    allowed=False,
                    reason="Rate limit exceeded",
                    action_taken=RuleAction.DENY,
                )

        # Check IP rules (fast path)
        if ip in self._ip_whitelist:
            return FilterResult(allowed=True, reason="IP whitelisted")

        if ip in self._ip_blacklist:
            self._record_violation(ip, "IP blacklisted")
            return FilterResult(
                allowed=False,
                reason="IP blacklisted",
                action_taken=RuleAction.DENY,
            )

        # Check CIDR rules
        try:
            ip_addr = ipaddress.ip_address(ip)
            for network, action in self._cidr_rules:
                if ip_addr in network:
                    if action == RuleAction.DENY:
                        self._record_violation(ip, f"CIDR blacklisted: {network}")
                        return FilterResult(
                            allowed=False,
                            reason=f"CIDR blacklisted: {network}",
                            action_taken=RuleAction.DENY,
                        )
                    elif action == RuleAction.ALLOW:
                        return FilterResult(
                            allowed=True, reason=f"CIDR whitelisted: {network}"
                        )
        except ValueError:
            pass

        # Check port rules
        if context.port:
            port_result = self._check_port(context.port, ip)
            if port_result:
                return port_result

        # Check path rules
        if context.path:
            path_result = self._check_path(context.path, ip)
            if path_result:
                return path_result

        # Default action
        if self._config.default_action == RuleAction.DENY:
            return FilterResult(
                allowed=False,
                reason="Default deny",
                action_taken=RuleAction.DENY,
            )

        return FilterResult(allowed=True, reason="Default allow")

    def _check_port(self, port: int, ip: str) -> Optional[FilterResult]:
        """Check port-based rules."""
        port_rules = [
            r
            for r in self._rules.values()
            if r.rule_type == RuleType.PORT and r.enabled and not r.is_expired()
        ]

        for rule in sorted(port_rules, key=lambda r: r.priority.value, reverse=True):
            if rule.value == str(port):
                rule.hit_count += 1
                rule.last_hit = datetime.now(timezone.utc)

                if rule.action == RuleAction.DENY:
                    self._record_violation(ip, f"Port denied: {port}")
                    return FilterResult(
                        allowed=False,
                        matched_rule=rule,
                        reason=f"Port denied: {port}",
                        action_taken=RuleAction.DENY,
                    )
                elif rule.action == RuleAction.ALLOW:
                    return FilterResult(
                        allowed=True,
                        matched_rule=rule,
                        reason=f"Port allowed: {port}",
                    )

        return None

    def _check_path(self, path: str, ip: str) -> Optional[FilterResult]:
        """Check path-based rules."""
        path_rules = [
            r
            for r in self._rules.values()
            if r.rule_type == RuleType.PATH and r.enabled and not r.is_expired()
        ]

        for rule in sorted(path_rules, key=lambda r: r.priority.value, reverse=True):
            try:
                if re.match(rule.value, path):
                    rule.hit_count += 1
                    rule.last_hit = datetime.now(timezone.utc)

                    if rule.action == RuleAction.DENY:
                        self._record_violation(ip, f"Path denied: {path}")
                        return FilterResult(
                            allowed=False,
                            matched_rule=rule,
                            reason=f"Path denied: {path}",
                            action_taken=RuleAction.DENY,
                        )
                    elif rule.action == RuleAction.ALLOW:
                        return FilterResult(
                            allowed=True,
                            matched_rule=rule,
                            reason=f"Path allowed: {path}",
                        )
            except re.error:
                logger.error(f"Invalid path pattern: {rule.value}")

        return None

    def _is_rate_limited(self, ip: str) -> bool:
        """Check if IP is rate limited."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self._config.rate_limit_window_seconds)

        # Get request history
        if ip not in self._request_counts:
            self._request_counts[ip] = []

        # Clean old entries
        self._request_counts[ip] = [
            ts for ts in self._request_counts[ip] if ts > window_start
        ]

        # Check limit
        if len(self._request_counts[ip]) >= self._config.rate_limit_max_requests:
            return True

        # Record this request
        self._request_counts[ip].append(now)
        return False

    def _record_violation(self, ip: str, reason: str) -> None:
        """Record a security violation."""
        self._violations[ip] = self._violations.get(ip, 0) + 1

        if self._config.log_denied:
            logger.warning(f"Firewall violation from {ip}: {reason}")

        # Check auto-ban threshold
        if (
            self._config.enable_auto_ban
            and self._violations[ip] >= self._config.auto_ban_threshold
        ):
            self._auto_ban(ip)

        # Notify callbacks
        for callback in self._on_violation:
            try:
                result = callback(ip, reason)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")

    def _auto_ban(self, ip: str) -> None:
        """Auto-ban an IP."""
        expires = datetime.now(timezone.utc) + timedelta(
            minutes=self._config.auto_ban_duration_minutes
        )
        self._auto_bans[ip] = expires
        logger.warning(
            f"Auto-banned {ip} for {self._config.auto_ban_duration_minutes} minutes"
        )

    def unban_ip(self, ip: str) -> bool:
        """Manually unban an IP."""
        if ip in self._auto_bans:
            del self._auto_bans[ip]
            self._violations.pop(ip, None)
            logger.info(f"Unbanned {ip}")
            return True
        return False

    def get_banned_ips(self) -> Dict[str, datetime]:
        """Get all currently banned IPs and their expiry times."""
        now = datetime.now(timezone.utc)
        return {ip: exp for ip, exp in self._auto_bans.items() if exp > now}

    def get_violations(self, ip: Optional[str] = None) -> Dict[str, int]:
        """Get violation counts."""
        if ip:
            return {ip: self._violations.get(ip, 0)}
        return dict(self._violations)

    def on_violation(
        self,
        callback: Callable[[str, str], Any],
    ) -> Callable[[], None]:
        """
        Register callback for security violations.

        Args:
            callback: Callback function(ip, reason)

        Returns:
            Unsubscribe function
        """
        self._on_violation.append(callback)

        def unsubscribe():
            if callback in self._on_violation:
                self._on_violation.remove(callback)

        return unsubscribe

    def get_stats(self) -> Dict[str, Any]:
        """Get firewall statistics."""
        return {
            "total_rules": len(self._rules),
            "active_rules": len(
                [r for r in self._rules.values() if r.enabled and not r.is_expired()]
            ),
            "whitelisted_ips": len(self._ip_whitelist),
            "blacklisted_ips": len(self._ip_blacklist),
            "cidr_rules": len(self._cidr_rules),
            "banned_ips": len(self.get_banned_ips()),
            "total_violations": sum(self._violations.values()),
        }
