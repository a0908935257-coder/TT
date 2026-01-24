"""
Structured logging module.

Provides JSON-formatted logging with correlation IDs, context injection,
and separate log channels for trading, audit, and performance logs.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# Context variables for correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
bot_id_var: ContextVar[Optional[str]] = ContextVar("bot_id", default=None)

F = TypeVar("F", bound=Callable[..., Any])


class LogChannel(Enum):
    """Log channels for different purposes."""

    APPLICATION = "application"
    TRADING = "trading"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    SECURITY = "security"


class EventType(Enum):
    """Standard event types for structured logging."""

    # Application events
    APP_START = "app.start"
    APP_STOP = "app.stop"
    APP_ERROR = "app.error"

    # Trading events
    ORDER_CREATED = "order.created"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_ERROR = "order.error"
    TRADE_EXECUTED = "trade.executed"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # Strategy events
    SIGNAL_GENERATED = "signal.generated"
    STRATEGY_START = "strategy.start"
    STRATEGY_STOP = "strategy.stop"

    # Risk events
    RISK_CHECK_PASSED = "risk.check_passed"
    RISK_CHECK_FAILED = "risk.check_failed"
    RISK_LIMIT_BREACH = "risk.limit_breach"
    STOP_LOSS_TRIGGERED = "stop_loss.triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit.triggered"

    # Audit events
    CONFIG_CHANGED = "config.changed"
    USER_ACTION = "user.action"
    PERMISSION_CHANGED = "permission.changed"
    API_KEY_USED = "api_key.used"

    # Performance events
    LATENCY_RECORDED = "latency.recorded"
    THROUGHPUT_RECORDED = "throughput.recorded"
    RESOURCE_USAGE = "resource.usage"

    # Security events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    RATE_LIMIT_HIT = "rate_limit.hit"
    SUSPICIOUS_ACTIVITY = "suspicious.activity"


@dataclass
class LogContext:
    """Context information for structured logs."""

    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    bot_id: Optional[str] = None
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    strategy: Optional[str] = None
    exchange: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "extra":
                    result.update(value)
                else:
                    result[key] = value
        return result


@dataclass
class StructuredLogRecord:
    """Structured log record for JSON output."""

    timestamp: str
    level: str
    channel: str
    event_type: str
    message: str
    logger_name: str
    context: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "channel": self.channel,
            "event_type": self.event_type,
            "message": self.message,
            "logger": self.logger_name,
            "context": self.context,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON structured logs."""

    def __init__(self, channel: LogChannel = LogChannel.APPLICATION):
        super().__init__()
        self._channel = channel

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get context from context variables
        context = {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "bot_id": bot_id_var.get(),
        }
        # Remove None values
        context = {k: v for k, v in context.items() if v is not None}

        # Get extra context from record
        if hasattr(record, "context") and isinstance(record.context, dict):
            context.update(record.context)

        # Get event type
        event_type = getattr(record, "event_type", "log.message")
        if isinstance(event_type, EventType):
            event_type = event_type.value

        # Get additional data
        data = getattr(record, "data", {})
        if not isinstance(data, dict):
            data = {"value": data}

        # Build structured record
        structured = StructuredLogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            channel=self._channel.value,
            event_type=event_type,
            message=record.getMessage(),
            logger_name=record.name,
            context=context,
            data=data,
        )

        return structured.to_json()


class StructuredLogger:
    """
    Structured logger with JSON output and correlation ID support.

    Provides separate log channels and rich context injection.
    """

    def __init__(
        self,
        name: str,
        channel: LogChannel = LogChannel.APPLICATION,
        level: int = logging.INFO,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            channel: Log channel
            level: Log level
            log_dir: Directory for log files
        """
        self._name = name
        self._channel = channel
        self._logger = logging.getLogger(f"structured.{channel.value}.{name}")
        self._logger.setLevel(level)
        self._logger.propagate = False

        # Set up log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # JSON file handler
        log_file = log_dir / f"{channel.value}.jsonl"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter(channel))
        self._logger.addHandler(file_handler)

        # Console handler for development
        if os.getenv("LOG_JSON_CONSOLE", "false").lower() == "true":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JSONFormatter(channel))
            self._logger.addHandler(console_handler)

    def _log(
        self,
        level: int,
        event_type: Union[str, EventType],
        message: str,
        context: Optional[LogContext] = None,
        **data: Any,
    ) -> None:
        """Internal logging method."""
        extra = {
            "event_type": event_type,
            "data": data,
        }
        if context:
            extra["context"] = context.to_dict()

        self._logger.log(level, message, extra=extra)

    def debug(
        self,
        event_type: Union[str, EventType],
        message: str,
        context: Optional[LogContext] = None,
        **data: Any,
    ) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, event_type, message, context, **data)

    def info(
        self,
        event_type: Union[str, EventType],
        message: str,
        context: Optional[LogContext] = None,
        **data: Any,
    ) -> None:
        """Log info message."""
        self._log(logging.INFO, event_type, message, context, **data)

    def warning(
        self,
        event_type: Union[str, EventType],
        message: str,
        context: Optional[LogContext] = None,
        **data: Any,
    ) -> None:
        """Log warning message."""
        self._log(logging.WARNING, event_type, message, context, **data)

    def error(
        self,
        event_type: Union[str, EventType],
        message: str,
        context: Optional[LogContext] = None,
        **data: Any,
    ) -> None:
        """Log error message."""
        self._log(logging.ERROR, event_type, message, context, **data)

    def critical(
        self,
        event_type: Union[str, EventType],
        message: str,
        context: Optional[LogContext] = None,
        **data: Any,
    ) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, event_type, message, context, **data)


class TradingLogger(StructuredLogger):
    """Specialized logger for trading events."""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        super().__init__(name, LogChannel.TRADING, log_dir=log_dir)

    def order_created(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        **extra: Any,
    ) -> None:
        """Log order creation."""
        ctx = LogContext(order_id=order_id, symbol=symbol)
        self.info(
            EventType.ORDER_CREATED,
            f"Order created: {side} {quantity} {symbol}",
            context=ctx,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            **extra,
        )

    def order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_quantity: float,
        fill_price: float,
        **extra: Any,
    ) -> None:
        """Log order fill."""
        ctx = LogContext(order_id=order_id, symbol=symbol)
        self.info(
            EventType.ORDER_FILLED,
            f"Order filled: {side} {filled_quantity} {symbol} @ {fill_price}",
            context=ctx,
            side=side,
            filled_quantity=filled_quantity,
            fill_price=fill_price,
            **extra,
        )

    def order_cancelled(
        self,
        order_id: str,
        symbol: str,
        reason: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log order cancellation."""
        ctx = LogContext(order_id=order_id, symbol=symbol)
        self.info(
            EventType.ORDER_CANCELLED,
            f"Order cancelled: {order_id}",
            context=ctx,
            reason=reason,
            **extra,
        )

    def order_rejected(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        **extra: Any,
    ) -> None:
        """Log order rejection."""
        ctx = LogContext(order_id=order_id, symbol=symbol)
        self.warning(
            EventType.ORDER_REJECTED,
            f"Order rejected: {order_id} - {reason}",
            context=ctx,
            reason=reason,
            **extra,
        )

    def trade_executed(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log trade execution."""
        ctx = LogContext(symbol=symbol)
        self.info(
            EventType.TRADE_EXECUTED,
            f"Trade executed: {side} {quantity} {symbol} @ {price}",
            context=ctx,
            trade_id=trade_id,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            **extra,
        )

    def signal_generated(
        self,
        strategy: str,
        symbol: str,
        signal: str,
        confidence: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log signal generation."""
        ctx = LogContext(strategy=strategy, symbol=symbol)
        self.info(
            EventType.SIGNAL_GENERATED,
            f"Signal generated: {signal} for {symbol}",
            context=ctx,
            signal=signal,
            confidence=confidence,
            **extra,
        )


class AuditLogger(StructuredLogger):
    """Specialized logger for audit events."""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        super().__init__(name, LogChannel.AUDIT, log_dir=log_dir)

    def config_changed(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        changed_by: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log configuration change."""
        ctx = LogContext(user_id=changed_by)
        self.info(
            EventType.CONFIG_CHANGED,
            f"Config changed: {config_key}",
            context=ctx,
            config_key=config_key,
            old_value=str(old_value),
            new_value=str(new_value),
            **extra,
        )

    def user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        target: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log user action."""
        ctx = LogContext(user_id=user_id)
        self.info(
            EventType.USER_ACTION,
            f"User action: {action}",
            context=ctx,
            action=action,
            target=target,
            **extra,
        )

    def api_key_used(
        self,
        key_id: str,
        endpoint: str,
        success: bool,
        **extra: Any,
    ) -> None:
        """Log API key usage."""
        self.info(
            EventType.API_KEY_USED,
            f"API key used: {key_id[:8]}... on {endpoint}",
            key_id_prefix=key_id[:8],
            endpoint=endpoint,
            success=success,
            **extra,
        )


class PerformanceLogger(StructuredLogger):
    """Specialized logger for performance events."""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        super().__init__(name, LogChannel.PERFORMANCE, log_dir=log_dir)

    def latency(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        **extra: Any,
    ) -> None:
        """Log latency measurement."""
        self.info(
            EventType.LATENCY_RECORDED,
            f"Latency: {operation} = {latency_ms:.2f}ms",
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            **extra,
        )

    def throughput(
        self,
        operation: str,
        requests_per_second: float,
        **extra: Any,
    ) -> None:
        """Log throughput measurement."""
        self.info(
            EventType.THROUGHPUT_RECORDED,
            f"Throughput: {operation} = {requests_per_second:.2f} req/s",
            operation=operation,
            requests_per_second=requests_per_second,
            **extra,
        )

    def resource_usage(
        self,
        cpu_percent: float,
        memory_mb: float,
        disk_percent: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log resource usage."""
        self.info(
            EventType.RESOURCE_USAGE,
            f"Resources: CPU={cpu_percent:.1f}%, Memory={memory_mb:.1f}MB",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_percent=disk_percent,
            **extra,
        )


class SecurityLogger(StructuredLogger):
    """Specialized logger for security events."""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        super().__init__(name, LogChannel.SECURITY, log_dir=log_dir)

    def auth_success(
        self,
        user_id: str,
        method: str,
        ip_address: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log successful authentication."""
        ctx = LogContext(user_id=user_id)
        self.info(
            EventType.AUTH_SUCCESS,
            f"Auth success: {user_id} via {method}",
            context=ctx,
            method=method,
            ip_address=ip_address,
            **extra,
        )

    def auth_failure(
        self,
        user_id: Optional[str],
        method: str,
        reason: str,
        ip_address: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log failed authentication."""
        ctx = LogContext(user_id=user_id)
        self.warning(
            EventType.AUTH_FAILURE,
            f"Auth failure: {user_id or 'unknown'} via {method} - {reason}",
            context=ctx,
            method=method,
            reason=reason,
            ip_address=ip_address,
            **extra,
        )

    def rate_limit_hit(
        self,
        source: str,
        limit: int,
        window_seconds: int,
        **extra: Any,
    ) -> None:
        """Log rate limit hit."""
        self.warning(
            EventType.RATE_LIMIT_HIT,
            f"Rate limit hit: {source} exceeded {limit} requests in {window_seconds}s",
            source=source,
            limit=limit,
            window_seconds=window_seconds,
            **extra,
        )

    def suspicious_activity(
        self,
        activity_type: str,
        description: str,
        severity: str = "medium",
        **extra: Any,
    ) -> None:
        """Log suspicious activity."""
        level = logging.WARNING if severity != "high" else logging.ERROR
        self._log(
            level,
            EventType.SUSPICIOUS_ACTIVITY,
            f"Suspicious activity: {activity_type} - {description}",
            activity_type=activity_type,
            severity=severity,
            **extra,
        )


# Context management functions
def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID in context, returns the ID used."""
    cid = correlation_id or generate_correlation_id()
    correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id_var.get()


def set_request_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    bot_id: Optional[str] = None,
) -> str:
    """Set request context, returns correlation ID."""
    cid = set_correlation_id(correlation_id)
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if bot_id:
        bot_id_var.set(bot_id)
    return cid


def clear_request_context() -> None:
    """Clear all request context."""
    correlation_id_var.set(None)
    request_id_var.set(None)
    user_id_var.set(None)
    bot_id_var.set(None)


def with_correlation_id(func: F) -> F:
    """Decorator to ensure correlation ID is set."""

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not get_correlation_id():
                set_correlation_id()
            return await func(*args, **kwargs)

        return async_wrapper  # type: ignore
    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not get_correlation_id():
                set_correlation_id()
            return func(*args, **kwargs)

        return sync_wrapper  # type: ignore


# Logger factory functions
_loggers: Dict[str, StructuredLogger] = {}


def get_trading_logger(name: str = "trading") -> TradingLogger:
    """Get or create a trading logger."""
    key = f"trading.{name}"
    if key not in _loggers:
        _loggers[key] = TradingLogger(name)
    return _loggers[key]  # type: ignore


def get_audit_logger(name: str = "audit") -> AuditLogger:
    """Get or create an audit logger."""
    key = f"audit.{name}"
    if key not in _loggers:
        _loggers[key] = AuditLogger(name)
    return _loggers[key]  # type: ignore


def get_performance_logger(name: str = "performance") -> PerformanceLogger:
    """Get or create a performance logger."""
    key = f"performance.{name}"
    if key not in _loggers:
        _loggers[key] = PerformanceLogger(name)
    return _loggers[key]  # type: ignore


def get_security_logger(name: str = "security") -> SecurityLogger:
    """Get or create a security logger."""
    key = f"security.{name}"
    if key not in _loggers:
        _loggers[key] = SecurityLogger(name)
    return _loggers[key]  # type: ignore
