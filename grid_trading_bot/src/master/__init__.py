"""
Master Module.

Bot registration center and lifecycle management.
"""

from .models import (
    BotAlreadyExistsError,
    BotInfo,
    BotNotFoundError,
    BotState,
    BotType,
    InvalidStateTransitionError,
    MarketType,
    RegistryEvent,
    VALID_STATE_TRANSITIONS,
)
from .registry import BotRegistry

__all__ = [
    # Models
    "BotInfo",
    "BotState",
    "BotType",
    "MarketType",
    "RegistryEvent",
    "VALID_STATE_TRANSITIONS",
    # Exceptions
    "BotAlreadyExistsError",
    "BotNotFoundError",
    "InvalidStateTransitionError",
    # Registry
    "BotRegistry",
]
