"""
Discord Bot Views.

Interactive components (buttons, dropdowns, modals) for the Discord bot.
"""

from .bot_controls import (
    BotControlView,
    BotControlWithDeleteView,
    DeleteButton,
    PauseButton,
    RefreshButton,
    ResumeButton,
    StartButton,
    StopButton,
    build_bot_embed,
    get_state_color,
    get_status_emoji,
)
from .bot_selector import (
    BotActionSelectorView,
    BotSelector,
    BotSelectorByState,
    BotSelectorView,
    MultipleBotsSelector,
)
from .confirm_views import (
    ConfirmDeleteView,
    ConfirmEmergencyView,
    ConfirmResetCircuitBreakerView,
    ConfirmStopAllView,
    ConfirmStopView,
)
from .create_modal import (
    CreateBotModal,
    EditBotModal,
    QuickCreateBotModal,
)

__all__ = [
    # Bot controls
    "BotControlView",
    "BotControlWithDeleteView",
    "StartButton",
    "StopButton",
    "PauseButton",
    "ResumeButton",
    "RefreshButton",
    "DeleteButton",
    "build_bot_embed",
    "get_status_emoji",
    "get_state_color",
    # Bot selectors
    "BotSelector",
    "BotSelectorByState",
    "MultipleBotsSelector",
    "BotSelectorView",
    "BotActionSelectorView",
    # Confirm views
    "ConfirmStopView",
    "ConfirmDeleteView",
    "ConfirmEmergencyView",
    "ConfirmStopAllView",
    "ConfirmResetCircuitBreakerView",
    # Modals
    "CreateBotModal",
    "QuickCreateBotModal",
    "EditBotModal",
]
