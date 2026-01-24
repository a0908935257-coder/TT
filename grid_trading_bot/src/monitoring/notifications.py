"""
Notification handlers module.

Provides multiple notification channels including Email, SMS, and integrations
with external services like PagerDuty and Telegram.
"""

import asyncio
import smtplib
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from src.core import get_logger
from src.monitoring.alerts import AlertChannelHandler, PersistedAlert

logger = get_logger(__name__)


@dataclass
class EmailConfig:
    """Email notification configuration."""

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    from_address: str = ""
    default_recipients: List[str] = field(default_factory=list)
    subject_prefix: str = "[Trading Alert]"


@dataclass
class SMSConfig:
    """SMS notification configuration (Twilio)."""

    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    default_recipients: List[str] = field(default_factory=list)


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""

    bot_token: str = ""
    default_chat_ids: List[str] = field(default_factory=list)


@dataclass
class PagerDutyConfig:
    """PagerDuty notification configuration."""

    routing_key: str = ""
    api_base: str = "https://events.pagerduty.com/v2/enqueue"


class EmailHandler(AlertChannelHandler):
    """
    Email notification handler.

    Sends alerts via SMTP email.
    """

    def __init__(self, config: EmailConfig):
        """
        Initialize email handler.

        Args:
            config: Email configuration
        """
        self._config = config

    async def send(self, alert: PersistedAlert) -> bool:
        """Send alert via email."""
        try:
            return await asyncio.to_thread(self._send_sync, alert)
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    def _send_sync(self, alert: PersistedAlert) -> bool:
        """Synchronous email sending."""
        if not self._config.username or not self._config.password:
            logger.warning("Email credentials not configured")
            return False

        recipients = alert.labels.get("recipients", "").split(",")
        if not recipients or not recipients[0]:
            recipients = self._config.default_recipients

        if not recipients:
            logger.warning("No email recipients configured")
            return False

        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"{self._config.subject_prefix} [{alert.severity.name}] {alert.title}"
        msg["From"] = self._config.from_address or self._config.username
        msg["To"] = ", ".join(recipients)

        # Plain text version
        text_body = self._format_text_body(alert)
        msg.attach(MIMEText(text_body, "plain"))

        # HTML version
        html_body = self._format_html_body(alert)
        msg.attach(MIMEText(html_body, "html"))

        # Send
        try:
            if self._config.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(
                    self._config.smtp_host, self._config.smtp_port
                ) as server:
                    server.starttls(context=context)
                    server.login(self._config.username, self._config.password)
                    server.sendmail(
                        self._config.from_address or self._config.username,
                        recipients,
                        msg.as_string(),
                    )
            else:
                with smtplib.SMTP(
                    self._config.smtp_host, self._config.smtp_port
                ) as server:
                    server.login(self._config.username, self._config.password)
                    server.sendmail(
                        self._config.from_address or self._config.username,
                        recipients,
                        msg.as_string(),
                    )

            logger.info(f"Email sent for alert {alert.alert_id}")
            return True

        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False

    def _format_text_body(self, alert: PersistedAlert) -> str:
        """Format plain text email body."""
        return f"""
Trading Alert: {alert.title}

Severity: {alert.severity.name}
State: {alert.state.value}
Source: {alert.source}
Time: {alert.fired_at.isoformat()}

Message:
{alert.message}

Labels: {alert.labels}
Annotations: {alert.annotations}

Alert ID: {alert.alert_id}
Fingerprint: {alert.fingerprint}
"""

    def _format_html_body(self, alert: PersistedAlert) -> str:
        """Format HTML email body."""
        severity_colors = {
            "DEBUG": "#6c757d",
            "INFO": "#17a2b8",
            "WARNING": "#ffc107",
            "ERROR": "#dc3545",
            "CRITICAL": "#721c24",
        }
        color = severity_colors.get(alert.severity.name, "#000000")

        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ padding: 20px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px; }}
        .label {{ font-weight: bold; }}
        .footer {{ margin-top: 20px; font-size: 12px; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>🚨 {alert.title}</h2>
        <p>Severity: {alert.severity.name} | State: {alert.state.value}</p>
    </div>
    <div class="content">
        <p><span class="label">Source:</span> {alert.source}</p>
        <p><span class="label">Time:</span> {alert.fired_at.isoformat()}</p>
        <p><span class="label">Message:</span></p>
        <p>{alert.message}</p>
    </div>
    <div class="footer">
        <p>Alert ID: {alert.alert_id} | Fingerprint: {alert.fingerprint}</p>
    </div>
</body>
</html>
"""


class SMSHandler(AlertChannelHandler):
    """
    SMS notification handler using Twilio.

    Sends alerts via SMS.
    """

    def __init__(self, config: SMSConfig):
        """
        Initialize SMS handler.

        Args:
            config: SMS configuration
        """
        self._config = config
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create Twilio client."""
        if self._client is None:
            try:
                from twilio.rest import Client

                self._client = Client(
                    self._config.account_sid, self._config.auth_token
                )
            except ImportError:
                logger.error("Twilio library not installed. Run: pip install twilio")
                raise
        return self._client

    async def send(self, alert: PersistedAlert) -> bool:
        """Send alert via SMS."""
        try:
            return await asyncio.to_thread(self._send_sync, alert)
        except Exception as e:
            logger.error(f"SMS notification failed: {e}")
            return False

    def _send_sync(self, alert: PersistedAlert) -> bool:
        """Synchronous SMS sending."""
        if not self._config.account_sid or not self._config.auth_token:
            logger.warning("Twilio credentials not configured")
            return False

        recipients = alert.labels.get("phone_numbers", "").split(",")
        if not recipients or not recipients[0]:
            recipients = self._config.default_recipients

        if not recipients:
            logger.warning("No SMS recipients configured")
            return False

        try:
            client = self._get_client()

            # Format message (SMS has character limits)
            message = self._format_sms(alert)

            # Send to each recipient
            success = True
            for recipient in recipients:
                recipient = recipient.strip()
                if not recipient:
                    continue

                try:
                    client.messages.create(
                        body=message,
                        from_=self._config.from_number,
                        to=recipient,
                    )
                    logger.info(f"SMS sent to {recipient} for alert {alert.alert_id}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {recipient}: {e}")
                    success = False

            return success

        except ImportError:
            return False
        except Exception as e:
            logger.error(f"SMS send error: {e}")
            return False

    def _format_sms(self, alert: PersistedAlert) -> str:
        """Format SMS message (max 160 chars for single SMS)."""
        # Keep it concise
        severity = alert.severity.name[0]  # First letter only
        message = f"[{severity}] {alert.title}: {alert.message}"

        # Truncate if needed
        if len(message) > 155:
            message = message[:152] + "..."

        return message


class TelegramHandler(AlertChannelHandler):
    """
    Telegram notification handler.

    Sends alerts via Telegram bot.
    """

    def __init__(self, config: TelegramConfig):
        """
        Initialize Telegram handler.

        Args:
            config: Telegram configuration
        """
        self._config = config
        self._api_base = f"https://api.telegram.org/bot{config.bot_token}"

    async def send(self, alert: PersistedAlert) -> bool:
        """Send alert via Telegram."""
        if not self._config.bot_token:
            logger.warning("Telegram bot token not configured")
            return False

        chat_ids = alert.labels.get("telegram_chat_ids", "").split(",")
        if not chat_ids or not chat_ids[0]:
            chat_ids = self._config.default_chat_ids

        if not chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return False

        try:
            import aiohttp

            message = self._format_message(alert)
            success = True

            async with aiohttp.ClientSession() as session:
                for chat_id in chat_ids:
                    chat_id = chat_id.strip()
                    if not chat_id:
                        continue

                    try:
                        async with session.post(
                            f"{self._api_base}/sendMessage",
                            json={
                                "chat_id": chat_id,
                                "text": message,
                                "parse_mode": "HTML",
                            },
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status != 200:
                                logger.error(
                                    f"Telegram API error: {await response.text()}"
                                )
                                success = False
                            else:
                                logger.info(
                                    f"Telegram sent to {chat_id} for alert {alert.alert_id}"
                                )
                    except Exception as e:
                        logger.error(f"Failed to send Telegram to {chat_id}: {e}")
                        success = False

            return success

        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")
            return False

    def _format_message(self, alert: PersistedAlert) -> str:
        """Format Telegram message with HTML."""
        severity_emoji = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨",
        }
        emoji = severity_emoji.get(alert.severity.name, "📢")

        return f"""
{emoji} <b>{alert.title}</b>

<b>Severity:</b> {alert.severity.name}
<b>Source:</b> {alert.source}
<b>Time:</b> {alert.fired_at.strftime('%Y-%m-%d %H:%M:%S')} UTC

{alert.message}

<i>Alert ID: {alert.alert_id}</i>
"""


class PagerDutyHandler(AlertChannelHandler):
    """
    PagerDuty notification handler.

    Sends alerts to PagerDuty for on-call escalation.
    """

    def __init__(self, config: PagerDutyConfig):
        """
        Initialize PagerDuty handler.

        Args:
            config: PagerDuty configuration
        """
        self._config = config

    async def send(self, alert: PersistedAlert) -> bool:
        """Send alert to PagerDuty."""
        if not self._config.routing_key:
            logger.warning("PagerDuty routing key not configured")
            return False

        try:
            import aiohttp

            # Map severity to PagerDuty severity
            pd_severity = self._map_severity(alert.severity.name)

            # Build PagerDuty event
            event = {
                "routing_key": self._config.routing_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": f"{alert.title}: {alert.message}",
                    "severity": pd_severity,
                    "source": alert.source,
                    "timestamp": alert.fired_at.isoformat(),
                    "custom_details": {
                        "alert_id": alert.alert_id,
                        "labels": alert.labels,
                        "annotations": alert.annotations,
                    },
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.api_base,
                    json=event,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty event sent for alert {alert.alert_id}")
                        return True
                    else:
                        body = await response.text()
                        logger.error(f"PagerDuty API error: {response.status} - {body}")
                        return False

        except Exception as e:
            logger.error(f"PagerDuty notification failed: {e}")
            return False

    def _map_severity(self, severity: str) -> str:
        """Map internal severity to PagerDuty severity."""
        mapping = {
            "DEBUG": "info",
            "INFO": "info",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
        }
        return mapping.get(severity, "warning")


class DiscordHandler(AlertChannelHandler):
    """
    Discord notification handler.

    Sends alerts via Discord webhook.
    """

    def __init__(self, webhook_url: str):
        """
        Initialize Discord handler.

        Args:
            webhook_url: Discord webhook URL
        """
        self._webhook_url = webhook_url

    async def send(self, alert: PersistedAlert) -> bool:
        """Send alert via Discord webhook."""
        if not self._webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        try:
            import aiohttp

            # Map severity to Discord embed color
            colors = {
                "DEBUG": 0x6C757D,
                "INFO": 0x17A2B8,
                "WARNING": 0xFFC107,
                "ERROR": 0xDC3545,
                "CRITICAL": 0x721C24,
            }
            color = colors.get(alert.severity.name, 0x000000)

            # Build Discord embed
            embed = {
                "title": f"🚨 {alert.title}",
                "description": alert.message,
                "color": color,
                "fields": [
                    {"name": "Severity", "value": alert.severity.name, "inline": True},
                    {"name": "Source", "value": alert.source, "inline": True},
                    {"name": "State", "value": alert.state.value, "inline": True},
                ],
                "footer": {"text": f"Alert ID: {alert.alert_id}"},
                "timestamp": alert.fired_at.isoformat(),
            }

            payload = {"embeds": [embed]}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Discord notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        body = await response.text()
                        logger.error(f"Discord webhook error: {response.status} - {body}")
                        return False

        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False


class NotificationManager:
    """
    Manages notification channels and delivery.

    Provides unified interface for sending alerts through multiple channels.
    """

    def __init__(self):
        """Initialize notification manager."""
        self._handlers: Dict[str, AlertChannelHandler] = {}

    def register_email(self, config: EmailConfig) -> None:
        """Register email notification handler."""
        self._handlers["email"] = EmailHandler(config)

    def register_sms(self, config: SMSConfig) -> None:
        """Register SMS notification handler."""
        self._handlers["sms"] = SMSHandler(config)

    def register_telegram(self, config: TelegramConfig) -> None:
        """Register Telegram notification handler."""
        self._handlers["telegram"] = TelegramHandler(config)

    def register_pagerduty(self, config: PagerDutyConfig) -> None:
        """Register PagerDuty notification handler."""
        self._handlers["pagerduty"] = PagerDutyHandler(config)

    def register_discord(self, webhook_url: str) -> None:
        """Register Discord notification handler."""
        self._handlers["discord"] = DiscordHandler(webhook_url)

    def get_handler(self, channel: str) -> Optional[AlertChannelHandler]:
        """Get handler for a channel."""
        return self._handlers.get(channel)

    async def send_to_channel(
        self, channel: str, alert: PersistedAlert
    ) -> bool:
        """Send alert to specific channel."""
        handler = self._handlers.get(channel)
        if not handler:
            logger.warning(f"No handler registered for channel: {channel}")
            return False

        return await handler.send(alert)

    async def send_to_all(self, alert: PersistedAlert) -> Dict[str, bool]:
        """Send alert to all registered channels."""
        results = {}
        for name, handler in self._handlers.items():
            try:
                results[name] = await handler.send(alert)
            except Exception as e:
                logger.error(f"Error sending to {name}: {e}")
                results[name] = False
        return results

    def list_channels(self) -> List[str]:
        """List all registered channels."""
        return list(self._handlers.keys())
