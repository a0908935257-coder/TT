"""
Discord Bot for Grid Trading Bot.

Provides interactive commands to check bot status and performance.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Callable, Any
from datetime import datetime, timezone

import discord
from discord.ext import commands

from core import get_logger

logger = get_logger(__name__)


class TradingDiscordBot:
    """
    Discord Bot for trading bot interaction.

    Supports commands:
        /status - Get current bot status
        /stats - Get performance statistics
        /orders - List active orders
        /balance - Check account balance
        /help - Show available commands
    """

    def __init__(self, token: str):
        """
        Initialize Discord Bot.

        Args:
            token: Discord bot token
        """
        self._token = token
        self._bot: Optional[commands.Bot] = None
        self._running = False

        # Callbacks to get data from trading bot
        self._get_status: Optional[Callable[[], dict]] = None
        self._get_statistics: Optional[Callable[[], dict]] = None
        self._get_orders: Optional[Callable[[], list]] = None
        self._get_balance: Optional[Callable[[], dict]] = None

        # Bot reference
        self._trading_bot: Any = None

    def set_trading_bot(self, trading_bot: Any) -> None:
        """Set reference to trading bot for data access."""
        self._trading_bot = trading_bot

    def set_callbacks(
        self,
        get_status: Optional[Callable[[], dict]] = None,
        get_statistics: Optional[Callable[[], dict]] = None,
        get_orders: Optional[Callable[[], list]] = None,
        get_balance: Optional[Callable[[], dict]] = None,
    ) -> None:
        """
        Set callback functions to get data.

        Args:
            get_status: Function returning bot status dict
            get_statistics: Function returning statistics dict
            get_orders: Function returning list of orders
            get_balance: Function returning balance dict
        """
        if get_status:
            self._get_status = get_status
        if get_statistics:
            self._get_statistics = get_statistics
        if get_orders:
            self._get_orders = get_orders
        if get_balance:
            self._get_balance = get_balance

    async def start(self) -> None:
        """Start the Discord bot."""
        if self._running:
            return

        # Create bot with intents
        intents = discord.Intents.default()
        intents.message_content = True

        self._bot = commands.Bot(command_prefix="!", intents=intents)

        # Remove default help command
        self._bot.remove_command('help')

        # Register event handlers
        @self._bot.event
        async def on_ready():
            logger.info(f"Discord Bot connected as {self._bot.user}")
            # Sync slash commands
            try:
                synced = await self._bot.tree.sync()
                logger.info(f"Synced {len(synced)} slash commands")
            except Exception as e:
                logger.error(f"Failed to sync commands: {e}")

        # Register slash commands
        @self._bot.tree.command(name="status", description="查看機器人狀態")
        async def status_command(interaction: discord.Interaction):
            await self._handle_status(interaction)

        @self._bot.tree.command(name="stats", description="查看績效統計")
        async def stats_command(interaction: discord.Interaction):
            await self._handle_stats(interaction)

        @self._bot.tree.command(name="orders", description="查看掛單")
        async def orders_command(interaction: discord.Interaction):
            await self._handle_orders(interaction)

        @self._bot.tree.command(name="balance", description="查看餘額")
        async def balance_command(interaction: discord.Interaction):
            await self._handle_balance(interaction)

        @self._bot.tree.command(name="help", description="顯示指令說明")
        async def help_command(interaction: discord.Interaction):
            await self._handle_help(interaction)

        # Also support ! prefix commands
        @self._bot.command(name="status")
        async def status_prefix(ctx):
            await self._send_status(ctx)

        @self._bot.command(name="stats")
        async def stats_prefix(ctx):
            await self._send_stats(ctx)

        @self._bot.command(name="orders")
        async def orders_prefix(ctx):
            await self._send_orders(ctx)

        @self._bot.command(name="balance")
        async def balance_prefix(ctx):
            await self._send_balance(ctx)

        @self._bot.command(name="help")
        async def help_prefix(ctx):
            await self._send_help(ctx)

        self._running = True

        # Start bot in background
        try:
            await self._bot.start(self._token)
        except Exception as e:
            logger.error(f"Discord bot error: {e}")
            self._running = False

    async def stop(self) -> None:
        """Stop the Discord bot."""
        if self._bot and self._running:
            await self._bot.close()
            self._running = False
            logger.info("Discord Bot stopped")

    # =========================================================================
    # Slash Command Handlers
    # =========================================================================

    async def _handle_status(self, interaction: discord.Interaction) -> None:
        """Handle /status command."""
        embed = self._create_status_embed()
        await interaction.response.send_message(embed=embed)

    async def _handle_stats(self, interaction: discord.Interaction) -> None:
        """Handle /stats command."""
        embed = self._create_stats_embed()
        await interaction.response.send_message(embed=embed)

    async def _handle_orders(self, interaction: discord.Interaction) -> None:
        """Handle /orders command."""
        embed = self._create_orders_embed()
        await interaction.response.send_message(embed=embed)

    async def _handle_balance(self, interaction: discord.Interaction) -> None:
        """Handle /balance command."""
        embed = self._create_balance_embed()
        await interaction.response.send_message(embed=embed)

    async def _handle_help(self, interaction: discord.Interaction) -> None:
        """Handle /help command."""
        embed = self._create_help_embed()
        await interaction.response.send_message(embed=embed)

    # =========================================================================
    # Prefix Command Handlers
    # =========================================================================

    async def _send_status(self, ctx) -> None:
        """Send status via prefix command."""
        embed = self._create_status_embed()
        await ctx.send(embed=embed)

    async def _send_stats(self, ctx) -> None:
        """Send stats via prefix command."""
        embed = self._create_stats_embed()
        await ctx.send(embed=embed)

    async def _send_orders(self, ctx) -> None:
        """Send orders via prefix command."""
        embed = self._create_orders_embed()
        await ctx.send(embed=embed)

    async def _send_balance(self, ctx) -> None:
        """Send balance via prefix command."""
        embed = self._create_balance_embed()
        await ctx.send(embed=embed)

    async def _send_help(self, ctx) -> None:
        """Send help via prefix command."""
        embed = self._create_help_embed()
        await ctx.send(embed=embed)

    # =========================================================================
    # Embed Creators
    # =========================================================================

    def _create_status_embed(self) -> discord.Embed:
        """Create status embed."""
        embed = discord.Embed(
            title="🤖 機器人狀態",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc),
        )

        if self._trading_bot:
            try:
                state = getattr(self._trading_bot, '_state', None)
                config = getattr(self._trading_bot, '_config', None)
                setup = getattr(self._trading_bot, '_setup', None)

                status = state.value if state else "未知"
                embed.add_field(name="狀態", value=status, inline=True)

                if config:
                    embed.add_field(name="交易對", value=config.symbol, inline=True)
                    embed.add_field(name="投資金額", value=f"{config.total_investment} USDT", inline=True)

                if setup:
                    embed.add_field(
                        name="網格範圍",
                        value=f"${float(setup.lower_price):.2f} - ${float(setup.upper_price):.2f}",
                        inline=False
                    )
                    embed.add_field(name="網格數量", value=f"{setup.grid_count} 格", inline=True)

            except Exception as e:
                embed.add_field(name="錯誤", value=str(e), inline=False)
        else:
            embed.add_field(name="狀態", value="未連接到交易機器人", inline=False)

        return embed

    def _create_stats_embed(self) -> discord.Embed:
        """Create statistics embed."""
        embed = discord.Embed(
            title="📊 績效統計",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        if self._trading_bot:
            try:
                stats = self._trading_bot.get_statistics()

                total_profit = stats.get('total_profit', Decimal('0'))
                trade_count = stats.get('trade_count', 0)
                avg_profit = stats.get('avg_profit_per_trade', Decimal('0'))
                total_fees = stats.get('total_fees', Decimal('0'))

                # Profit color
                profit_emoji = "🟢" if total_profit >= 0 else "🔴"

                embed.add_field(
                    name=f"{profit_emoji} 總利潤",
                    value=f"{float(total_profit):.4f} USDT",
                    inline=True
                )
                embed.add_field(name="交易次數", value=str(trade_count), inline=True)
                embed.add_field(
                    name="平均每筆利潤",
                    value=f"{float(avg_profit):.4f} USDT",
                    inline=True
                )
                embed.add_field(
                    name="總手續費",
                    value=f"{float(total_fees):.4f} USDT",
                    inline=True
                )

                # Pending orders
                pending_buy = stats.get('pending_buy_count', 0)
                pending_sell = stats.get('pending_sell_count', 0)
                embed.add_field(name="待成交買單", value=str(pending_buy), inline=True)
                embed.add_field(name="待成交賣單", value=str(pending_sell), inline=True)

            except Exception as e:
                embed.add_field(name="錯誤", value=str(e), inline=False)
        else:
            embed.add_field(name="狀態", value="未連接到交易機器人", inline=False)

        return embed

    def _create_orders_embed(self) -> discord.Embed:
        """Create orders embed."""
        embed = discord.Embed(
            title="📋 掛單列表",
            color=discord.Color.orange(),
            timestamp=datetime.now(timezone.utc),
        )

        if self._trading_bot:
            try:
                order_manager = getattr(self._trading_bot, '_order_manager', None)
                if order_manager:
                    orders = order_manager._orders

                    if not orders:
                        embed.add_field(name="掛單", value="目前沒有掛單", inline=False)
                    else:
                        buy_orders = []
                        sell_orders = []

                        for order_id, order in orders.items():
                            order_str = f"`{float(order.quantity):.5f}` @ `${float(order.price):.2f}`"
                            if order.side.value == "BUY":
                                buy_orders.append(order_str)
                            else:
                                sell_orders.append(order_str)

                        if buy_orders:
                            embed.add_field(
                                name=f"🟢 買單 ({len(buy_orders)})",
                                value="\n".join(buy_orders[:5]) or "無",
                                inline=True
                            )
                        if sell_orders:
                            embed.add_field(
                                name=f"🔴 賣單 ({len(sell_orders)})",
                                value="\n".join(sell_orders[:5]) or "無",
                                inline=True
                            )
                else:
                    embed.add_field(name="狀態", value="無法取得訂單資訊", inline=False)

            except Exception as e:
                embed.add_field(name="錯誤", value=str(e), inline=False)
        else:
            embed.add_field(name="狀態", value="未連接到交易機器人", inline=False)

        return embed

    def _create_balance_embed(self) -> discord.Embed:
        """Create balance embed."""
        embed = discord.Embed(
            title="💰 帳戶餘額",
            color=discord.Color.gold(),
            timestamp=datetime.now(timezone.utc),
        )

        if self._trading_bot:
            try:
                # This would need async call, simplified version
                embed.add_field(
                    name="提示",
                    value="餘額查詢需要連接交易所\n請使用 /stats 查看績效",
                    inline=False
                )
            except Exception as e:
                embed.add_field(name="錯誤", value=str(e), inline=False)
        else:
            embed.add_field(name="狀態", value="未連接到交易機器人", inline=False)

        return embed

    def _create_help_embed(self) -> discord.Embed:
        """Create help embed."""
        embed = discord.Embed(
            title="📖 指令說明",
            description="網格交易機器人 Discord 指令",
            color=discord.Color.purple(),
        )

        commands_info = [
            ("/status", "查看機器人狀態"),
            ("/stats", "查看績效統計"),
            ("/orders", "查看目前掛單"),
            ("/balance", "查看帳戶餘額"),
            ("/help", "顯示此說明"),
        ]

        for cmd, desc in commands_info:
            embed.add_field(name=cmd, value=desc, inline=False)

        embed.set_footer(text="也可以使用 ! 前綴，例如 !status")

        return embed
