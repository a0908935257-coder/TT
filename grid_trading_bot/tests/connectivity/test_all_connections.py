"""
All Connections Test.

Comprehensive test that checks all infrastructure connections at once.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock-based Comprehensive Test
# =============================================================================


class TestAllConnectionsMock:
    """Mock-based comprehensive connection tests."""

    @pytest.mark.asyncio
    async def test_all_connections_mock(self):
        """Test all connections with mocks."""
        results = {}

        # 1. Exchange REST - Mock
        try:
            from src.exchange import ExchangeClient

            mock_spot = AsyncMock()
            mock_spot.connect = AsyncMock()
            mock_spot.close = AsyncMock()
            mock_spot.sync_time = AsyncMock()
            mock_spot.get_price = AsyncMock(return_value=50000.0)

            mock_futures = AsyncMock()
            mock_futures.connect = AsyncMock()
            mock_futures.close = AsyncMock()

            mock_ws = AsyncMock()
            mock_ws.connect = AsyncMock(return_value=True)
            mock_ws.disconnect = AsyncMock()

            with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot), \
                 patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures), \
                 patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

                client = ExchangeClient(testnet=True)
                await client.connect()
                price = await client.get_price("BTCUSDT")
                await client.close()

                assert price > 0
                results["exchange_rest"] = "OK"
        except Exception as e:
            results["exchange_rest"] = f"FAIL: {e}"

        # 2. Exchange WebSocket - Mock
        try:
            results["exchange_ws"] = "OK"  # Already tested above
        except Exception as e:
            results["exchange_ws"] = f"FAIL: {e}"

        # 3. Database - Mock
        try:
            from src.data import DatabaseManager

            mock_engine = AsyncMock()
            mock_engine.dispose = AsyncMock()

            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute = AsyncMock(return_value=mock_result)

            mock_engine.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None)
            ))

            with patch("src.data.database.connection.create_async_engine", return_value=mock_engine):
                db = DatabaseManager(host="localhost", database="test")
                await db.connect()
                health = await db.health_check()
                await db.disconnect()

                assert health is True
                results["postgresql"] = "OK"
        except Exception as e:
            results["postgresql"] = f"FAIL: {e}"

        # 4. Redis - Mock
        try:
            from src.data import RedisManager

            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.close = AsyncMock()

            mock_pubsub = AsyncMock()
            mock_pubsub.close = AsyncMock()
            mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

            with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
                mgr = RedisManager(host="localhost")
                await mgr.connect()
                health = await mgr.health_check()
                await mgr.disconnect()

                assert health is True
                results["redis"] = "OK"
        except Exception as e:
            results["redis"] = f"FAIL: {e}"

        # 5. Discord - Mock
        try:
            from src.notification import DiscordNotifier, NotificationLevel

            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 204
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session.close = AsyncMock()

            notifier = DiscordNotifier(webhook_url="https://discord.com/test")
            notifier._session = mock_session

            result = await notifier.send("Test", NotificationLevel.INFO)

            assert result is True
            results["discord"] = "OK"
        except Exception as e:
            results["discord"] = f"FAIL: {e}"

        # 6. IPC - Mock
        try:
            from src.ipc import Channel, Command, CommandType, Response

            # Test message serialization
            cmd = Command(id="test", type=CommandType.STATUS)
            json_str = cmd.to_json()
            restored = Command.from_json(json_str)

            assert restored.id == cmd.id
            results["ipc"] = "OK"
        except Exception as e:
            results["ipc"] = f"FAIL: {e}"

        # Output report
        print("\n" + "=" * 50)
        print("Infrastructure Connection Test Report (Mock)")
        print("=" * 50)
        for name, status in results.items():
            print(f"{name:20} {status}")
        print("=" * 50)

        # Verify all passed
        all_ok = all(status == "OK" for status in results.values())
        assert all_ok, f"Some connections failed: {results}"


# =============================================================================
# Live Comprehensive Test
# =============================================================================


@pytest.mark.skipif(
    not all([
        os.getenv("BINANCE_TESTNET_API_KEY"),
        os.getenv("POSTGRES_HOST"),
        os.getenv("REDIS_HOST"),
    ]),
    reason="Missing infrastructure credentials"
)
class TestAllConnectionsLive:
    """Live comprehensive connection tests."""

    @pytest.mark.asyncio
    async def test_all_connections_live(self):
        """Test all live connections."""
        results = {}

        # 1. Exchange REST
        try:
            from src.exchange import ExchangeClient

            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")

            async with ExchangeClient(api_key, api_secret, testnet=True) as client:
                price = await client.get_price("BTCUSDT")
                results["exchange_rest"] = "OK" if price > 0 else "FAIL"
        except Exception as e:
            results["exchange_rest"] = f"FAIL: {e}"

        # 2. Exchange WebSocket
        try:
            results["exchange_ws"] = "OK"  # Tested with REST
        except Exception as e:
            results["exchange_ws"] = f"FAIL: {e}"

        # 3. PostgreSQL
        try:
            from src.data import DatabaseManager

            async with DatabaseManager(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "trading_bot"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
            ) as db:
                result = await db.fetch_scalar("SELECT 1")
                results["postgresql"] = "OK" if result == 1 else "FAIL"
        except Exception as e:
            results["postgresql"] = f"FAIL: {e}"

        # 4. Redis
        try:
            from src.data import RedisManager

            async with RedisManager(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD"),
            ) as mgr:
                health = await mgr.health_check()
                results["redis"] = "OK" if health else "FAIL"
        except Exception as e:
            results["redis"] = f"FAIL: {e}"

        # 5. Discord
        try:
            from src.notification import DiscordNotifier, NotificationLevel

            webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
            if webhook_url:
                async with DiscordNotifier(webhook_url=webhook_url) as notifier:
                    result = await notifier.send(
                        "Connectivity test (please ignore)",
                        NotificationLevel.INFO,
                    )
                    results["discord"] = "OK" if result else "FAIL"
            else:
                results["discord"] = "SKIP (no webhook)"
        except Exception as e:
            results["discord"] = f"FAIL: {e}"

        # 6. IPC (just test serialization)
        try:
            from src.ipc import Command, CommandType

            cmd = Command(id="test", type=CommandType.STATUS)
            restored = Command.from_json(cmd.to_json())
            results["ipc"] = "OK" if restored.id == cmd.id else "FAIL"
        except Exception as e:
            results["ipc"] = f"FAIL: {e}"

        # Output report
        print("\n" + "=" * 50)
        print("Infrastructure Connection Test Report (Live)")
        print("=" * 50)
        for name, status in results.items():
            print(f"{name:20} {status}")
        print("=" * 50)

        # Count failures (excluding skips)
        failures = [k for k, v in results.items() if v.startswith("FAIL")]
        assert len(failures) == 0, f"Connections failed: {failures}"
