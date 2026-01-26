"""
Core module for Trading Master.

Provides logging utilities and timeout mechanisms.
"""

from .logger import setup_logger, get_logger
from .structured_logging import (
    # Loggers
    StructuredLogger,
    TradingLogger,
    AuditLogger,
    PerformanceLogger,
    SecurityLogger,
    # Logger factories
    get_trading_logger,
    get_audit_logger,
    get_performance_logger,
    get_security_logger,
    # Types
    LogChannel,
    EventType,
    LogContext,
    # Context management
    set_correlation_id,
    get_correlation_id,
    set_request_context,
    clear_request_context,
    with_correlation_id,
)
from .timeout import (
    TimeoutError,
    TimeoutConfig,
    TimeoutContext,
    get_timeout_config,
    set_timeout_config,
    with_timeout,
    with_timeout_and_cancel,
    with_retry_timeout,
    command_timeout,
    fund_timeout,
    notification_timeout,
)

__all__ = [
    # Basic logging
    "setup_logger",
    "get_logger",
    # Structured logging
    "StructuredLogger",
    "TradingLogger",
    "AuditLogger",
    "PerformanceLogger",
    "SecurityLogger",
    "get_trading_logger",
    "get_audit_logger",
    "get_performance_logger",
    "get_security_logger",
    "LogChannel",
    "EventType",
    "LogContext",
    "set_correlation_id",
    "get_correlation_id",
    "set_request_context",
    "clear_request_context",
    "with_correlation_id",
    # Timeout utilities
    "TimeoutError",
    "TimeoutConfig",
    "TimeoutContext",
    "get_timeout_config",
    "set_timeout_config",
    "with_timeout",
    "with_timeout_and_cancel",
    "with_retry_timeout",
    "command_timeout",
    "fund_timeout",
    "notification_timeout",
]
