"""
Discord Bot Tests.

Test suite for the Discord trading bot.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockRole:
    """Mock Discord Role."""

    id: int
    name: str = "Test Role"


@dataclass
class MockGuildPermissions:
    """Mock Discord Guild Permissions."""

    administrator: bool = False


@dataclass
class MockMember:
    """Mock Discord Member."""

    id: int
    name: str = "TestUser"
    roles: List[MockRole] = field(default_factory=list)
    guild_permissions: MockGuildPermissions = field(default_factory=MockGuildPermissions)

    @property
    def display_name(self) -> str:
        return self.name


@dataclass
class MockUser:
    """Mock Discord User (for DMs)."""

    id: int
    name: str = "TestUser"


@dataclass
class MockChannel:
    """Mock Discord Channel."""

    id: int
    name: str = "test-channel"

    async def send(self, content: str = None, embed=None, view=None) -> "MockMessage":
        return MockMessage(content=content, embed=embed)


@dataclass
class MockMessage:
    """Mock Discord Message."""

    content: Optional[str] = None
    embed: Any = None


@dataclass
class MockGuild:
    """Mock Discord Guild."""

    id: int
    name: str = "Test Guild"
    owner_id: int = 12345


class MockResponse:
    """Mock Discord Interaction Response."""

    def __init__(self):
        self._is_done = False
        self._deferred = False

    def is_done(self) -> bool:
        return self._is_done

    async def defer(self, ephemeral: bool = False):
        self._deferred = True
        self._is_done = True

    async def send_message(self, content: str = None, embed=None, view=None, ephemeral: bool = False):
        self._is_done = True
        return MockMessage(content=content, embed=embed)

    async def send_modal(self, modal):
        self._is_done = True


class MockFollowup:
    """Mock Discord Interaction Followup."""

    def __init__(self):
        self.messages = []

    async def send(self, content: str = None, embed=None, embeds=None, view=None, ephemeral: bool = False):
        msg = MockMessage(content=content, embed=embed)
        self.messages.append(msg)
        return msg


class MockInteraction:
    """Mock Discord Interaction."""

    def __init__(
        self,
        user: Optional[MockMember] = None,
        channel: Optional[MockChannel] = None,
        guild: Optional[MockGuild] = None,
        client: Any = None,
    ):
        self.user = user or MockMember(id=12345)
        self.channel = channel or MockChannel(id=67890)
        self.channel_id = self.channel.id
        self.guild = guild or MockGuild(id=11111)
        self.guild_id = self.guild.id
        self.client = client or MagicMock()
        self.response = MockResponse()
        self.followup = MockFollowup()

    async def edit_original_response(self, content: str = None, embed=None, view=None):
        return MockMessage(content=content, embed=embed)


# =============================================================================
# Mock Bot State
# =============================================================================


class MockBotState:
    """Mock Bot State enum."""

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value


class MockBotType:
    """Mock Bot Type enum."""

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value


@dataclass
class MockBotInfo:
    """Mock Bot Info."""

    bot_id: str
    symbol: str
    state: MockBotState
    bot_type: MockBotType = field(default_factory=lambda: MockBotType("grid"))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    profit: Optional[Decimal] = None


# =============================================================================
# Mock Result
# =============================================================================


@dataclass
class MockResult:
    """Mock operation result."""

    success: bool
    message: str = ""
    bot_id: Optional[str] = None
    data: Optional[Dict] = None


# =============================================================================
# Mock Master
# =============================================================================


class MockMaster:
    """Mock Master class for testing."""

    def __init__(self, bots: Optional[List[MockBotInfo]] = None):
        self._bots = bots or []
        self.registry = MagicMock()

    def get_all_bots(self) -> List[MockBotInfo]:
        return self._bots

    def get_bot(self, bot_id: str) -> Optional[MockBotInfo]:
        for bot in self._bots:
            if bot.bot_id == bot_id:
                return bot
        return None

    def get_dashboard_data(self):
        """Return mock dashboard data."""
        return MagicMock(
            summary=MagicMock(
                total_bots=len(self._bots),
                running_bots=sum(1 for b in self._bots if b.state.value == "running"),
                paused_bots=sum(1 for b in self._bots if b.state.value == "paused"),
                error_bots=sum(1 for b in self._bots if b.state.value == "error"),
                total_investment=Decimal("10000"),
                total_value=Decimal("10500"),
                total_profit=Decimal("500"),
                total_profit_rate=Decimal("0.05"),
                today_profit=Decimal("50"),
                today_trades=10,
            ),
            bots=self._bots,
            alerts=[],
        )

    async def create_bot(self, bot_type, config: Dict) -> MockResult:
        bot_id = f"bot_{len(self._bots) + 1}"
        new_bot = MockBotInfo(
            bot_id=bot_id,
            symbol=config.get("symbol", "BTCUSDT"),
            state=MockBotState("registered"),
        )
        self._bots.append(new_bot)
        return MockResult(success=True, bot_id=bot_id)

    async def start_bot(self, bot_id: str) -> MockResult:
        bot = self.get_bot(bot_id)
        if bot:
            bot.state = MockBotState("running")
            return MockResult(success=True, message="Started")
        return MockResult(success=False, message="Bot not found")

    async def stop_bot(self, bot_id: str) -> MockResult:
        bot = self.get_bot(bot_id)
        if bot:
            bot.state = MockBotState("stopped")
            return MockResult(success=True, message="Stopped", data={"total_trades": 10, "total_profit": 0.5})
        return MockResult(success=False, message="Bot not found")

    async def pause_bot(self, bot_id: str) -> MockResult:
        bot = self.get_bot(bot_id)
        if bot:
            bot.state = MockBotState("paused")
            return MockResult(success=True, message="Paused")
        return MockResult(success=False, message="Bot not found")

    async def resume_bot(self, bot_id: str) -> MockResult:
        bot = self.get_bot(bot_id)
        if bot:
            bot.state = MockBotState("running")
            return MockResult(success=True, message="Resumed")
        return MockResult(success=False, message="Bot not found")

    async def delete_bot(self, bot_id: str) -> MockResult:
        bot = self.get_bot(bot_id)
        if bot:
            self._bots.remove(bot)
            return MockResult(success=True, message="Deleted")
        return MockResult(success=False, message="Bot not found")


# =============================================================================
# Mock Risk Engine
# =============================================================================


class MockRiskEngine:
    """Mock Risk Engine for testing."""

    def __init__(self, level: str = "NORMAL"):
        self._level = level
        self._circuit_breaker_triggered = False

    def get_status(self):
        """Return mock risk status."""
        level_mock = MagicMock()
        level_mock.name = self._level
        return MagicMock(
            level=level_mock,
            capital=MagicMock(
                total_capital=Decimal("100000"),
                initial_capital=Decimal("100000"),
                available_balance=Decimal("50000"),
            ),
            drawdown=MagicMock(
                drawdown_pct=Decimal("0.05"),
                max_drawdown_pct=Decimal("0.08"),
                peak_value=Decimal("105000"),
            ),
            circuit_breaker=MagicMock(
                is_triggered=self._circuit_breaker_triggered,
                trigger_reason="Test",
                cooldown_until=None,
            ),
            daily_pnl=MagicMock(
                pnl=Decimal("500"),
                pnl_pct=Decimal("0.005"),
            ),
            active_alerts=[],
            statistics=MagicMock(
                total_checks=100,
                violations=5,
                circuit_breaker_triggers=0,
            ),
        )

    async def trigger_emergency(self, reason: str):
        self._circuit_breaker_triggered = True

    async def reset_circuit_breaker(self, force: bool = False) -> bool:
        self._circuit_breaker_triggered = False
        return True


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_bots(count: int = 3) -> List[MockBotInfo]:
    """Create a list of test bots."""
    states = ["running", "paused", "stopped"]
    bots = []
    for i in range(count):
        bots.append(
            MockBotInfo(
                bot_id=f"bot_{i + 1}",
                symbol=f"{'BTC' if i == 0 else 'ETH' if i == 1 else 'SOL'}USDT",
                state=MockBotState(states[i % len(states)]),
                profit=Decimal(str(10 * (i + 1))),
            )
        )
    return bots


def create_admin_member(user_id: int = 12345, admin_role_id: int = 99999) -> MockMember:
    """Create a mock admin member."""
    return MockMember(
        id=user_id,
        name="AdminUser",
        roles=[MockRole(id=admin_role_id, name="Admin")],
        guild_permissions=MockGuildPermissions(administrator=True),
    )


def create_user_member(user_id: int = 54321, user_role_id: int = 88888) -> MockMember:
    """Create a mock regular user member."""
    return MockMember(
        id=user_id,
        name="RegularUser",
        roles=[MockRole(id=user_role_id, name="User")],
        guild_permissions=MockGuildPermissions(administrator=False),
    )
