"""
Report generation module.

Provides automated report generation for trading performance,
system health, and operational metrics.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.core import get_logger

logger = get_logger(__name__)


class ReportPeriod(Enum):
    """Report time periods."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


@dataclass
class ReportConfig:
    """Report generation configuration."""

    period: ReportPeriod = ReportPeriod.DAILY
    format: ReportFormat = ReportFormat.HTML
    output_dir: Optional[Path] = None
    auto_email: bool = False
    email_recipients: List[str] = field(default_factory=list)
    include_charts: bool = True
    timezone_offset: int = 0  # Hours from UTC


@dataclass
class TradingMetrics:
    """Trading performance metrics for reports."""

    period_start: datetime
    period_end: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    avg_trade_duration: float = 0.0  # seconds
    sharpe_ratio: Optional[Decimal] = None
    total_volume: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": str(self.total_pnl),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "gross_profit": str(self.gross_profit),
            "gross_loss": str(self.gross_loss),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "win_rate": str(self.win_rate),
            "profit_factor": str(self.profit_factor),
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "largest_win": str(self.largest_win),
            "largest_loss": str(self.largest_loss),
            "avg_trade_duration": self.avg_trade_duration,
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio else None,
            "total_volume": str(self.total_volume),
            "total_fees": str(self.total_fees),
        }


@dataclass
class SystemMetrics:
    """System health metrics for reports."""

    period_start: datetime
    period_end: datetime
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_requests: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    uptime_percent: float = 0.0
    connection_count: int = 0
    rate_limit_hits: int = 0
    reconnection_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "max_cpu_percent": round(self.max_cpu_percent, 2),
            "avg_memory_mb": round(self.avg_memory_mb, 2),
            "max_memory_mb": round(self.max_memory_mb, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "uptime_percent": round(self.uptime_percent, 2),
            "connection_count": self.connection_count,
            "rate_limit_hits": self.rate_limit_hits,
            "reconnection_count": self.reconnection_count,
        }


@dataclass
class AlertMetrics:
    """Alert metrics for reports."""

    period_start: datetime
    period_end: datetime
    total_alerts: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)
    alerts_by_source: Dict[str, int] = field(default_factory=dict)
    avg_resolution_time_seconds: float = 0.0
    unresolved_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_alerts": self.total_alerts,
            "alerts_by_severity": self.alerts_by_severity,
            "alerts_by_source": self.alerts_by_source,
            "avg_resolution_time_seconds": round(self.avg_resolution_time_seconds, 2),
            "unresolved_count": self.unresolved_count,
        }


@dataclass
class Report:
    """Complete report model."""

    report_id: str
    period: ReportPeriod
    generated_at: datetime
    trading: Optional[TradingMetrics] = None
    system: Optional[SystemMetrics] = None
    alerts: Optional[AlertMetrics] = None
    custom_sections: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "period": self.period.value,
            "generated_at": self.generated_at.isoformat(),
            "trading": self.trading.to_dict() if self.trading else None,
            "system": self.system.to_dict() if self.system else None,
            "alerts": self.alerts.to_dict() if self.alerts else None,
            "custom_sections": self.custom_sections,
        }


class MetricsProvider(Protocol):
    """Protocol for metrics data providers."""

    async def get_trading_metrics(
        self, start: datetime, end: datetime
    ) -> TradingMetrics:
        """Get trading metrics for period."""
        ...

    async def get_system_metrics(
        self, start: datetime, end: datetime
    ) -> SystemMetrics:
        """Get system metrics for period."""
        ...

    async def get_alert_metrics(
        self, start: datetime, end: datetime
    ) -> AlertMetrics:
        """Get alert metrics for period."""
        ...


class ReportFormatter(ABC):
    """Abstract base for report formatters."""

    @abstractmethod
    def format(self, report: Report) -> str:
        """Format report to string."""
        pass


class JSONReportFormatter(ReportFormatter):
    """JSON report formatter."""

    def format(self, report: Report) -> str:
        """Format report as JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str)


class MarkdownReportFormatter(ReportFormatter):
    """Markdown report formatter."""

    def format(self, report: Report) -> str:
        """Format report as Markdown."""
        lines = [
            f"# Trading Report - {report.period.value.title()}",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
        ]

        if report.trading:
            t = report.trading
            lines.extend(
                [
                    "## Trading Performance",
                    "",
                    f"**Period:** {t.period_start.strftime('%Y-%m-%d %H:%M')} to {t.period_end.strftime('%Y-%m-%d %H:%M')}",
                    "",
                    "### Summary",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Total P&L | {t.total_pnl:,.2f} |",
                    f"| Win Rate | {float(t.win_rate):.1%} |",
                    f"| Profit Factor | {t.profit_factor:.2f} |",
                    f"| Total Trades | {t.total_trades} |",
                    f"| Max Drawdown | {float(t.max_drawdown_pct):.2%} |",
                    "",
                    "### Trade Breakdown",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Winning Trades | {t.winning_trades} |",
                    f"| Losing Trades | {t.losing_trades} |",
                    f"| Average Win | {t.avg_win:,.2f} |",
                    f"| Average Loss | {t.avg_loss:,.2f} |",
                    f"| Largest Win | {t.largest_win:,.2f} |",
                    f"| Largest Loss | {t.largest_loss:,.2f} |",
                    "",
                ]
            )

        if report.system:
            s = report.system
            lines.extend(
                [
                    "## System Health",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Uptime | {s.uptime_percent:.1%} |",
                    f"| Avg CPU | {s.avg_cpu_percent:.1f}% |",
                    f"| Max CPU | {s.max_cpu_percent:.1f}% |",
                    f"| Avg Memory | {s.avg_memory_mb:.0f} MB |",
                    f"| Avg Latency | {s.avg_latency_ms:.2f} ms |",
                    f"| P99 Latency | {s.p99_latency_ms:.2f} ms |",
                    f"| Total Requests | {s.total_requests:,} |",
                    f"| Error Rate | {s.error_rate:.2%} |",
                    "",
                ]
            )

        if report.alerts:
            a = report.alerts
            lines.extend(
                [
                    "## Alerts Summary",
                    "",
                    f"| Metric | Value |",
                    f"|--------|-------|",
                    f"| Total Alerts | {a.total_alerts} |",
                    f"| Unresolved | {a.unresolved_count} |",
                    f"| Avg Resolution Time | {a.avg_resolution_time_seconds/60:.1f} min |",
                    "",
                    "### By Severity",
                    "",
                ]
            )
            for severity, count in a.alerts_by_severity.items():
                lines.append(f"- **{severity}**: {count}")
            lines.append("")

        lines.extend(
            [
                "---",
                f"*Report ID: {report.report_id}*",
            ]
        )

        return "\n".join(lines)


class HTMLReportFormatter(ReportFormatter):
    """HTML report formatter."""

    def format(self, report: Report) -> str:
        """Format report as HTML."""
        trading_section = ""
        if report.trading:
            t = report.trading
            trading_section = f"""
            <section class="trading">
                <h2>📈 Trading Performance</h2>
                <div class="period">
                    {t.period_start.strftime('%Y-%m-%d %H:%M')} to {t.period_end.strftime('%Y-%m-%d %H:%M')}
                </div>
                <div class="metrics-grid">
                    <div class="metric-card {'positive' if float(t.total_pnl) >= 0 else 'negative'}">
                        <div class="metric-value">{t.total_pnl:,.2f}</div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{float(t.win_rate)*100:.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{t.profit_factor:.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{t.total_trades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                </div>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Winning Trades</td><td>{t.winning_trades}</td></tr>
                    <tr><td>Losing Trades</td><td>{t.losing_trades}</td></tr>
                    <tr><td>Average Win</td><td>{t.avg_win:,.2f}</td></tr>
                    <tr><td>Average Loss</td><td>{t.avg_loss:,.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td>{float(t.max_drawdown_pct)*100:.2f}%</td></tr>
                    <tr><td>Total Volume</td><td>{t.total_volume:,.2f}</td></tr>
                    <tr><td>Total Fees</td><td>{t.total_fees:,.2f}</td></tr>
                </table>
            </section>
            """

        system_section = ""
        if report.system:
            s = report.system
            system_section = f"""
            <section class="system">
                <h2>🖥️ System Health</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{s.uptime_percent:.1f}%</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{s.avg_latency_ms:.2f}ms</div>
                        <div class="metric-label">Avg Latency</div>
                    </div>
                    <div class="metric-card {'warning' if s.error_rate > 0.01 else ''}">
                        <div class="metric-value">{s.error_rate*100:.2f}%</div>
                        <div class="metric-label">Error Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{s.total_requests:,}</div>
                        <div class="metric-label">Total Requests</div>
                    </div>
                </div>
                <table>
                    <tr><th>Metric</th><th>Average</th><th>Peak</th></tr>
                    <tr><td>CPU Usage</td><td>{s.avg_cpu_percent:.1f}%</td><td>{s.max_cpu_percent:.1f}%</td></tr>
                    <tr><td>Memory</td><td>{s.avg_memory_mb:.0f} MB</td><td>{s.max_memory_mb:.0f} MB</td></tr>
                    <tr><td>Latency</td><td>{s.avg_latency_ms:.2f}ms</td><td>{s.p99_latency_ms:.2f}ms (p99)</td></tr>
                </table>
            </section>
            """

        alerts_section = ""
        if report.alerts:
            a = report.alerts
            severity_rows = "".join(
                f"<tr><td>{sev}</td><td>{cnt}</td></tr>"
                for sev, cnt in a.alerts_by_severity.items()
            )
            alerts_section = f"""
            <section class="alerts">
                <h2>🚨 Alerts Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{a.total_alerts}</div>
                        <div class="metric-label">Total Alerts</div>
                    </div>
                    <div class="metric-card {'warning' if a.unresolved_count > 0 else ''}">
                        <div class="metric-value">{a.unresolved_count}</div>
                        <div class="metric-label">Unresolved</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{a.avg_resolution_time_seconds/60:.1f}m</div>
                        <div class="metric-label">Avg Resolution</div>
                    </div>
                </div>
                <table>
                    <tr><th>Severity</th><th>Count</th></tr>
                    {severity_rows}
                </table>
            </section>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trading Report - {report.period.value.title()}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        header h1 {{ margin: 0 0 10px; }}
        header .date {{ opacity: 0.8; }}
        section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        section:last-child {{ border-bottom: none; }}
        h2 {{ color: #1a1a2e; margin-top: 0; }}
        .period {{
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card.positive {{ background: #d4edda; }}
        .metric-card.negative {{ background: #f8d7da; }}
        .metric-card.warning {{ background: #fff3cd; }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1a1a2e;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        footer {{
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Trading Report</h1>
            <div class="date">
                {report.period.value.title()} Report - Generated {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC
            </div>
        </header>
        {trading_section}
        {system_section}
        {alerts_section}
        <footer>
            Report ID: {report.report_id}
        </footer>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """
    Generates automated reports for trading operations.

    Supports multiple formats and scheduling.
    """

    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        metrics_provider: Optional[MetricsProvider] = None,
    ):
        """
        Initialize report generator.

        Args:
            config: Report configuration
            metrics_provider: Provider for metrics data
        """
        self._config = config or ReportConfig()
        self._metrics_provider = metrics_provider
        self._formatters: Dict[ReportFormat, ReportFormatter] = {
            ReportFormat.JSON: JSONReportFormatter(),
            ReportFormat.HTML: HTMLReportFormatter(),
            ReportFormat.MARKDOWN: MarkdownReportFormatter(),
        }

        # Set up output directory
        if self._config.output_dir is None:
            self._config.output_dir = Path(__file__).parent.parent.parent / "reports"
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def set_metrics_provider(self, provider: MetricsProvider) -> None:
        """Set the metrics provider."""
        self._metrics_provider = provider

    def _get_period_range(
        self,
        period: ReportPeriod,
        reference_time: Optional[datetime] = None,
    ) -> tuple[datetime, datetime]:
        """Get start and end times for a period."""
        now = reference_time or datetime.now(timezone.utc)

        if period == ReportPeriod.HOURLY:
            end = now.replace(minute=0, second=0, microsecond=0)
            start = end - timedelta(hours=1)
        elif period == ReportPeriod.DAILY:
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start = end - timedelta(days=1)
        elif period == ReportPeriod.WEEKLY:
            # Start from Monday
            days_since_monday = now.weekday()
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = end - timedelta(days=days_since_monday)
            start = end - timedelta(weeks=1)
        elif period == ReportPeriod.MONTHLY:
            end = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Go to previous month
            if end.month == 1:
                start = end.replace(year=end.year - 1, month=12)
            else:
                start = end.replace(month=end.month - 1)
        else:
            # Custom - default to last 24 hours
            end = now
            start = now - timedelta(days=1)

        return start, end

    async def generate(
        self,
        period: Optional[ReportPeriod] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        include_trading: bool = True,
        include_system: bool = True,
        include_alerts: bool = True,
    ) -> Report:
        """
        Generate a report.

        Args:
            period: Report period (uses config default if not specified)
            start_time: Custom start time
            end_time: Custom end time
            include_trading: Include trading metrics
            include_system: Include system metrics
            include_alerts: Include alert metrics

        Returns:
            Generated Report object
        """
        period = period or self._config.period

        # Determine time range
        if start_time and end_time:
            start, end = start_time, end_time
        else:
            start, end = self._get_period_range(period)

        # Generate report ID
        report_id = f"{period.value}_{start.strftime('%Y%m%d_%H%M')}"

        # Collect metrics
        trading_metrics = None
        system_metrics = None
        alert_metrics = None

        if self._metrics_provider:
            if include_trading:
                try:
                    trading_metrics = await self._metrics_provider.get_trading_metrics(
                        start, end
                    )
                except Exception as e:
                    logger.error(f"Failed to get trading metrics: {e}")

            if include_system:
                try:
                    system_metrics = await self._metrics_provider.get_system_metrics(
                        start, end
                    )
                except Exception as e:
                    logger.error(f"Failed to get system metrics: {e}")

            if include_alerts:
                try:
                    alert_metrics = await self._metrics_provider.get_alert_metrics(
                        start, end
                    )
                except Exception as e:
                    logger.error(f"Failed to get alert metrics: {e}")
        else:
            # Create empty metrics with period info
            if include_trading:
                trading_metrics = TradingMetrics(period_start=start, period_end=end)
            if include_system:
                system_metrics = SystemMetrics(period_start=start, period_end=end)
            if include_alerts:
                alert_metrics = AlertMetrics(period_start=start, period_end=end)

        report = Report(
            report_id=report_id,
            period=period,
            generated_at=datetime.now(timezone.utc),
            trading=trading_metrics,
            system=system_metrics,
            alerts=alert_metrics,
        )

        return report

    def format(
        self,
        report: Report,
        format_type: Optional[ReportFormat] = None,
    ) -> str:
        """
        Format a report.

        Args:
            report: Report to format
            format_type: Output format (uses config default if not specified)

        Returns:
            Formatted report string
        """
        format_type = format_type or self._config.format
        formatter = self._formatters.get(format_type)

        if not formatter:
            raise ValueError(f"Unsupported format: {format_type}")

        return formatter.format(report)

    async def save(
        self,
        report: Report,
        format_type: Optional[ReportFormat] = None,
    ) -> Path:
        """
        Save a report to file.

        Args:
            report: Report to save
            format_type: Output format

        Returns:
            Path to saved file
        """
        format_type = format_type or self._config.format
        formatted = self.format(report, format_type)

        # Determine file extension
        extensions = {
            ReportFormat.JSON: ".json",
            ReportFormat.HTML: ".html",
            ReportFormat.MARKDOWN: ".md",
            ReportFormat.CSV: ".csv",
        }
        ext = extensions.get(format_type, ".txt")

        # Build filename
        filename = f"report_{report.report_id}{ext}"
        filepath = self._config.output_dir / filename

        # Save file
        filepath.write_text(formatted, encoding="utf-8")
        logger.info(f"Report saved: {filepath}")

        return filepath

    async def generate_and_save(
        self,
        period: Optional[ReportPeriod] = None,
        format_type: Optional[ReportFormat] = None,
    ) -> tuple[Report, Path]:
        """
        Generate and save a report.

        Args:
            period: Report period
            format_type: Output format

        Returns:
            Tuple of (Report, file path)
        """
        report = await self.generate(period)
        path = await self.save(report, format_type)
        return report, path


class ReportScheduler:
    """
    Schedules automated report generation.

    Runs reports on configured intervals.
    """

    def __init__(self, generator: ReportGenerator):
        """
        Initialize scheduler.

        Args:
            generator: Report generator instance
        """
        self._generator = generator
        self._running = False
        self._tasks: Dict[ReportPeriod, asyncio.Task] = {}

    async def start(self, periods: Optional[List[ReportPeriod]] = None) -> None:
        """Start scheduled report generation."""
        if self._running:
            return

        self._running = True
        periods = periods or [ReportPeriod.DAILY]

        for period in periods:
            self._tasks[period] = asyncio.create_task(self._run_schedule(period))

        logger.info(f"Report scheduler started for periods: {[p.value for p in periods]}")

    async def stop(self) -> None:
        """Stop scheduled report generation."""
        self._running = False

        for task in self._tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Report scheduler stopped")

    async def _run_schedule(self, period: ReportPeriod) -> None:
        """Run scheduled report for a period."""
        # Calculate intervals
        intervals = {
            ReportPeriod.HOURLY: 3600,
            ReportPeriod.DAILY: 86400,
            ReportPeriod.WEEKLY: 604800,
            ReportPeriod.MONTHLY: 2592000,  # ~30 days
        }
        interval = intervals.get(period, 86400)

        while self._running:
            try:
                # Wait until next scheduled time
                now = datetime.now(timezone.utc)

                if period == ReportPeriod.HOURLY:
                    # Run at the top of each hour
                    next_run = now.replace(minute=0, second=0, microsecond=0)
                    next_run += timedelta(hours=1)
                elif period == ReportPeriod.DAILY:
                    # Run at midnight UTC
                    next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    next_run += timedelta(days=1)
                else:
                    next_run = now + timedelta(seconds=interval)

                wait_seconds = (next_run - now).total_seconds()
                await asyncio.sleep(wait_seconds)

                # Generate and save report
                report, path = await self._generator.generate_and_save(period)
                logger.info(f"Scheduled {period.value} report generated: {path}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in report schedule ({period.value}): {e}")
                await asyncio.sleep(60)  # Wait before retry
