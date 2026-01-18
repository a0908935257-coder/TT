"""
Tests for Discord Bot Views.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.discord_bot.views import (
    BotControlView,
    BotSelector,
    BotSelectorView,
    ConfirmDeleteView,
    ConfirmEmergencyView,
    ConfirmStopView,
    CreateBotModal,
)
from tests.discord_bot import (
    MockBotInfo,
    MockBotState,
    MockInteraction,
    MockMaster,
    MockMember,
    MockRiskEngine,
    create_test_bots,
)


# =============================================================================
# BotControlView Tests
# =============================================================================


class TestBotControlView:
    """Tests for BotControlView."""

    @pytest.mark.asyncio
    async def test_stopped_bot_has_start_button(self):
        """Test stopped bot shows start button."""
        master = MockMaster()
        view = BotControlView("bot_1", "stopped", master)

        button_labels = [item.label for item in view.children if hasattr(item, "label")]
        assert "Start" in button_labels
        assert "Refresh" in button_labels
        assert "Stop" not in button_labels
        assert "Pause" not in button_labels

    @pytest.mark.asyncio
    async def test_running_bot_has_pause_stop_buttons(self):
        """Test running bot shows pause and stop buttons."""
        master = MockMaster()
        view = BotControlView("bot_1", "running", master)

        button_labels = [item.label for item in view.children if hasattr(item, "label")]
        assert "Pause" in button_labels
        assert "Stop" in button_labels
        assert "Refresh" in button_labels
        assert "Start" not in button_labels

    @pytest.mark.asyncio
    async def test_paused_bot_has_resume_stop_buttons(self):
        """Test paused bot shows resume and stop buttons."""
        master = MockMaster()
        view = BotControlView("bot_1", "paused", master)

        button_labels = [item.label for item in view.children if hasattr(item, "label")]
        assert "Resume" in button_labels
        assert "Stop" in button_labels
        assert "Refresh" in button_labels
        assert "Start" not in button_labels
        assert "Pause" not in button_labels

    @pytest.mark.asyncio
    async def test_view_timeout(self):
        """Test view has timeout."""
        master = MockMaster()
        view = BotControlView("bot_1", "running", master)
        assert view.timeout == 300


# =============================================================================
# BotSelector Tests
# =============================================================================


class TestBotSelector:
    """Tests for BotSelector."""

    @pytest.mark.asyncio
    async def test_selector_with_bots(self):
        """Test selector with bots."""
        bots = create_test_bots(3)
        selector = BotSelector(bots)

        assert len(selector.options) == 3
        assert selector.options[0].value == "bot_1"
        assert selector.options[1].value == "bot_2"
        assert selector.options[2].value == "bot_3"

    @pytest.mark.asyncio
    async def test_selector_empty_bots(self):
        """Test selector with no bots."""
        selector = BotSelector([])

        assert len(selector.options) == 1
        assert selector.options[0].value == "none"

    @pytest.mark.asyncio
    async def test_selector_max_25_bots(self):
        """Test selector limits to 25 bots (Discord limit)."""
        bots = [
            MockBotInfo(
                bot_id=f"bot_{i}",
                symbol="BTCUSDT",
                state=MockBotState("running"),
            )
            for i in range(30)
        ]
        selector = BotSelector(bots)

        assert len(selector.options) == 25

    @pytest.mark.asyncio
    async def test_selector_callback_with_none(self):
        """Test selector callback when no bots selected."""
        selector = BotSelector([])
        # Mock the values property since it's read-only
        with patch.object(type(selector), "values", new_callable=lambda: property(lambda self: ["none"])):
            interaction = MockInteraction()
            await selector.callback(interaction)

            assert interaction.response._is_done


class TestBotSelectorView:
    """Tests for BotSelectorView."""

    @pytest.mark.asyncio
    async def test_view_has_selector(self):
        """Test view contains selector."""
        bots = create_test_bots(3)
        master = MockMaster(bots)
        view = BotSelectorView(bots, master)

        # Should have one child (the selector)
        assert len(view.children) == 1


# =============================================================================
# Confirm Views Tests
# =============================================================================


class TestConfirmStopView:
    """Tests for ConfirmStopView."""

    @pytest.mark.asyncio
    async def test_view_has_confirm_and_cancel(self):
        """Test view has confirm and cancel buttons."""
        master = MockMaster()
        view = ConfirmStopView("bot_1", master)

        button_labels = [item.label for item in view.children if hasattr(item, "label")]
        assert "Confirm Stop" in button_labels
        assert "Cancel" in button_labels

    @pytest.mark.asyncio
    async def test_confirm_stops_bot(self):
        """Test confirm button stops bot."""
        bots = [
            MockBotInfo(
                bot_id="bot_1",
                symbol="BTCUSDT",
                state=MockBotState("running"),
            )
        ]
        master = MockMaster(bots)
        view = ConfirmStopView("bot_1", master)

        interaction = MockInteraction()

        # Find and call confirm button
        confirm_button = None
        for item in view.children:
            if hasattr(item, "label") and "Confirm" in item.label:
                confirm_button = item
                break

        assert confirm_button is not None
        await confirm_button.callback(interaction)

        # Check bot is stopped
        bot = master.get_bot("bot_1")
        assert bot.state.value == "stopped"


class TestConfirmDeleteView:
    """Tests for ConfirmDeleteView."""

    @pytest.mark.asyncio
    async def test_view_has_confirm_and_cancel(self):
        """Test view has confirm and cancel buttons."""
        master = MockMaster()
        view = ConfirmDeleteView("bot_1", master)

        button_labels = [item.label for item in view.children if hasattr(item, "label")]
        assert "Confirm Delete" in button_labels
        assert "Cancel" in button_labels

    @pytest.mark.asyncio
    async def test_confirm_deletes_bot(self):
        """Test confirm button deletes bot."""
        bots = [
            MockBotInfo(
                bot_id="bot_1",
                symbol="BTCUSDT",
                state=MockBotState("stopped"),
            )
        ]
        master = MockMaster(bots)
        view = ConfirmDeleteView("bot_1", master)

        interaction = MockInteraction()

        # Find and call confirm button
        confirm_button = None
        for item in view.children:
            if hasattr(item, "label") and "Confirm" in item.label:
                confirm_button = item
                break

        assert confirm_button is not None
        await confirm_button.callback(interaction)

        # Check bot is deleted
        assert master.get_bot("bot_1") is None


class TestConfirmEmergencyView:
    """Tests for ConfirmEmergencyView."""

    @pytest.mark.asyncio
    async def test_view_has_confirm_and_cancel(self):
        """Test view has confirm and cancel buttons."""
        risk_engine = MockRiskEngine()
        view = ConfirmEmergencyView(risk_engine)

        button_labels = [item.label for item in view.children if hasattr(item, "label")]
        assert any("EMERGENCY" in label.upper() for label in button_labels)
        assert "Cancel" in button_labels

    @pytest.mark.asyncio
    async def test_view_short_timeout(self):
        """Test view has short timeout for emergency."""
        risk_engine = MockRiskEngine()
        view = ConfirmEmergencyView(risk_engine)

        assert view.timeout == 30

    @pytest.mark.asyncio
    async def test_confirm_triggers_emergency(self):
        """Test confirm button triggers emergency."""
        risk_engine = MockRiskEngine()
        view = ConfirmEmergencyView(risk_engine)

        interaction = MockInteraction()

        # Find and call confirm button
        confirm_button = None
        for item in view.children:
            if hasattr(item, "label") and "EMERGENCY" in item.label.upper():
                confirm_button = item
                break

        assert confirm_button is not None
        await confirm_button.callback(interaction)

        # Check emergency was triggered
        assert risk_engine._circuit_breaker_triggered is True


# =============================================================================
# CreateBotModal Tests
# =============================================================================


class TestCreateBotModal:
    """Tests for CreateBotModal."""

    @pytest.mark.asyncio
    async def test_modal_has_required_fields(self):
        """Test modal has all required fields."""
        master = MockMaster()
        modal = CreateBotModal(master)

        # Check text inputs exist
        assert hasattr(modal, "symbol")
        assert hasattr(modal, "investment")
        assert hasattr(modal, "grid_count")
        assert hasattr(modal, "atr_multiplier")

    @pytest.mark.asyncio
    async def test_modal_creates_bot(self):
        """Test modal submit creates bot."""
        master = MockMaster()
        modal = CreateBotModal(master)

        # Mock the text inputs with proper values
        modal.symbol = MagicMock(value="ETHUSDT")
        modal.investment = MagicMock(value="500")
        modal.grid_count = MagicMock(value="15")
        modal.atr_multiplier = MagicMock(value="2.5")

        interaction = MockInteraction()
        await modal.on_submit(interaction)

        # Check bot was created
        assert len(master._bots) == 1
        assert master._bots[0].symbol == "ETHUSDT"

    @pytest.mark.asyncio
    async def test_modal_validates_symbol(self):
        """Test modal validates symbol."""
        master = MockMaster()
        modal = CreateBotModal(master)

        # Mock empty symbol
        modal.symbol = MagicMock(value="")
        modal.investment = MagicMock(value="500")
        modal.grid_count = MagicMock(value="15")
        modal.atr_multiplier = MagicMock(value="2.5")

        interaction = MockInteraction()
        await modal.on_submit(interaction)

        # Bot should not be created
        assert len(master._bots) == 0

    @pytest.mark.asyncio
    async def test_modal_validates_investment(self):
        """Test modal validates investment."""
        master = MockMaster()
        modal = CreateBotModal(master)

        # Mock invalid investment
        modal.symbol = MagicMock(value="BTCUSDT")
        modal.investment = MagicMock(value="invalid")
        modal.grid_count = MagicMock(value="15")
        modal.atr_multiplier = MagicMock(value="2.5")

        interaction = MockInteraction()
        await modal.on_submit(interaction)

        # Bot should not be created
        assert len(master._bots) == 0


# =============================================================================
# View Integration Tests
# =============================================================================


class TestViewIntegration:
    """Integration tests for views."""

    @pytest.mark.asyncio
    async def test_control_view_button_updates_view(self):
        """Test button click updates the view."""
        bots = [
            MockBotInfo(
                bot_id="bot_1",
                symbol="BTCUSDT",
                state=MockBotState("stopped"),
            )
        ]
        master = MockMaster(bots)
        view = BotControlView("bot_1", "stopped", master)

        # Find start button
        start_button = None
        for item in view.children:
            if hasattr(item, "label") and item.label == "Start":
                start_button = item
                break

        assert start_button is not None

        # Click start
        interaction = MockInteraction()
        await start_button.callback(interaction)

        # Bot should now be running
        bot = master.get_bot("bot_1")
        assert bot.state.value == "running"
