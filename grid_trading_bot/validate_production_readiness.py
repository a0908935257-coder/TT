#!/usr/bin/env python3
"""
Production Readiness Validation Script.

Validates that the trading system meets all requirements for live trading:
1. Core module imports
2. Configuration validation
3. External service connectivity
4. Exchange API connectivity
5. Monitoring system functionality
6. Alert system functionality
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment
os.environ.setdefault("ENVIRONMENT", "development")


class ValidationResult:
    """Result of a single validation check."""

    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
        self.duration_ms = 0.0

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f" - {self.message}" if self.message else ""
        return f"{status} | {self.name}{msg}"


class ProductionReadinessValidator:
    """Validates system readiness for production trading."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        print(result)

    async def run_all_validations(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 70)
        print("🔍 交易系統實戰資格驗證")
        print("=" * 70 + "\n")

        # Phase 1: Core Imports
        print("\n📦 Phase 1: 核心模組導入驗證")
        print("-" * 50)
        await self._validate_core_imports()

        # Phase 2: Configuration
        print("\n⚙️ Phase 2: 配置檔案驗證")
        print("-" * 50)
        await self._validate_configuration()

        # Phase 3: External Services
        print("\n🔌 Phase 3: 外部服務連接驗證")
        print("-" * 50)
        await self._validate_external_services()

        # Phase 4: Exchange Connectivity
        print("\n📊 Phase 4: 交易所 API 驗證")
        print("-" * 50)
        await self._validate_exchange()

        # Phase 5: Monitoring System
        print("\n📈 Phase 5: 監控系統驗證")
        print("-" * 50)
        await self._validate_monitoring()

        # Phase 6: Alert System
        print("\n🚨 Phase 6: 告警系統驗證")
        print("-" * 50)
        await self._validate_alerts()

        # Summary
        return self._print_summary()

    async def _validate_core_imports(self) -> None:
        """Validate all core modules can be imported."""
        modules_to_check = [
            ("src.core", "核心模組"),
            ("src.core.models", "數據模型"),
            ("src.exchange", "交易所客戶端"),
            ("src.bots.base", "機器人基類"),
            ("src.bots.grid", "Grid 策略"),
            ("src.monitoring", "監控系統"),
            ("src.monitoring.alerts", "告警系統"),
            ("src.monitoring.watchdog", "看門狗"),
            ("src.monitoring.comprehensive_monitor", "全面監控"),
            ("src.master", "主控台"),
            ("src.fund_manager", "資金管理"),
            ("src.config", "配置系統"),
            ("src.risk", "風險管理"),
        ]

        for module_path, description in modules_to_check:
            start = time.time()
            try:
                __import__(module_path)
                duration = (time.time() - start) * 1000
                result = ValidationResult(
                    name=f"Import {description}",
                    passed=True,
                    message=f"{module_path} ({duration:.1f}ms)",
                )
                result.duration_ms = duration
            except Exception as e:
                result = ValidationResult(
                    name=f"Import {description}",
                    passed=False,
                    message=f"{module_path}: {str(e)[:50]}",
                )
            self.add_result(result)

    async def _validate_configuration(self) -> None:
        """Validate configuration files."""
        # Check .env file
        env_file = Path(__file__).parent / ".env"
        result = ValidationResult(
            name="環境變數檔案",
            passed=env_file.exists(),
            message=str(env_file) if env_file.exists() else "Missing .env file",
        )
        self.add_result(result)

        # Check required environment variables
        required_vars = [
            ("BINANCE_API_KEY", "交易所 API Key"),
            ("BINANCE_API_SECRET", "交易所 API Secret"),
            ("DISCORD_WEBHOOK_URL", "Discord Webhook"),
        ]

        for var_name, description in required_vars:
            value = os.environ.get(var_name, "")
            passed = bool(value) and len(value) > 10
            result = ValidationResult(
                name=f"環境變數: {description}",
                passed=passed,
                message="已配置" if passed else "未配置或無效",
            )
            self.add_result(result)

        # Check config.yaml
        config_file = Path(__file__).parent / "config" / "config.yaml"
        result = ValidationResult(
            name="主配置檔案",
            passed=config_file.exists(),
            message=str(config_file) if config_file.exists() else "Missing config.yaml",
        )
        self.add_result(result)

        # Validate config loading
        try:
            from src.config import load_config
            config = load_config("config/config.yaml")
            result = ValidationResult(
                name="配置載入驗證",
                passed=True,
                message=f"Environment: {config.environment}",
            )
        except Exception as e:
            result = ValidationResult(
                name="配置載入驗證",
                passed=False,
                message=str(e)[:50],
            )
        self.add_result(result)

    async def _validate_external_services(self) -> None:
        """Validate external service connectivity."""
        # Redis
        try:
            import redis.asyncio as aioredis
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))

            start = time.time()
            client = aioredis.from_url(f"redis://{redis_host}:{redis_port}")
            await asyncio.wait_for(client.ping(), timeout=5.0)
            await client.close()
            duration = (time.time() - start) * 1000

            result = ValidationResult(
                name="Redis 連接",
                passed=True,
                message=f"{redis_host}:{redis_port} ({duration:.1f}ms)",
            )
        except asyncio.TimeoutError:
            result = ValidationResult(
                name="Redis 連接",
                passed=False,
                message="連接超時",
            )
        except Exception as e:
            result = ValidationResult(
                name="Redis 連接",
                passed=False,
                message=str(e)[:50],
            )
        self.add_result(result)

        # PostgreSQL
        try:
            import asyncpg
            db_host = os.environ.get("DB_HOST", "localhost")
            db_port = int(os.environ.get("DB_PORT", 5432))
            db_name = os.environ.get("DB_NAME", "trading_bot")
            db_user = os.environ.get("DB_USER", "postgres")
            db_pass = os.environ.get("DB_PASSWORD", "")

            start = time.time()
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=db_host,
                    port=db_port,
                    database=db_name,
                    user=db_user,
                    password=db_pass,
                ),
                timeout=5.0,
            )
            await conn.fetchval("SELECT 1")
            await conn.close()
            duration = (time.time() - start) * 1000

            result = ValidationResult(
                name="PostgreSQL 連接",
                passed=True,
                message=f"{db_host}:{db_port}/{db_name} ({duration:.1f}ms)",
            )
        except asyncio.TimeoutError:
            result = ValidationResult(
                name="PostgreSQL 連接",
                passed=False,
                message="連接超時",
            )
        except Exception as e:
            result = ValidationResult(
                name="PostgreSQL 連接",
                passed=False,
                message=str(e)[:50],
            )
        self.add_result(result)

    async def _validate_exchange(self) -> None:
        """Validate exchange API connectivity."""
        try:
            from src.exchange import ExchangeClient
            from src.config import load_config

            config = load_config("config/config.yaml")

            # Get API credentials from environment
            api_key = os.environ.get("BINANCE_API_KEY", "")
            api_secret = os.environ.get("BINANCE_API_SECRET", "")
            testnet = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"

            start = time.time()
            client = ExchangeClient(api_key, api_secret, testnet=testnet)
            await client.connect()

            # Test server time
            server_time = await asyncio.wait_for(
                client.spot.get_server_time(),
                timeout=10.0,
            )
            duration = (time.time() - start) * 1000

            result = ValidationResult(
                name="交易所時間同步",
                passed=True,
                message=f"Server time: {server_time} ({duration:.1f}ms)",
            )
            self.add_result(result)

            # Test account access (Futures - primary market for grid trading)
            from src.core.models import MarketType
            start = time.time()
            try:
                account = await asyncio.wait_for(
                    client.get_account(market=MarketType.FUTURES),
                    timeout=10.0,
                )
                duration = (time.time() - start) * 1000

                # Check balances
                usdt_balance = Decimal("0")
                balance = account.get_balance("USDT")
                if balance:
                    usdt_balance = balance.free

                result = ValidationResult(
                    name="交易所帳戶訪問 (Futures)",
                    passed=True,
                    message=f"USDT 餘額: {usdt_balance} ({duration:.1f}ms)",
                )
            except Exception as e:
                result = ValidationResult(
                    name="交易所帳戶訪問 (Futures)",
                    passed=False,
                    message=str(e)[:50],
                )
            self.add_result(result)

            # Test market data
            start = time.time()
            try:
                ticker = await asyncio.wait_for(
                    client.get_ticker("BTCUSDT"),
                    timeout=10.0,
                )
                duration = (time.time() - start) * 1000

                price = ticker.price
                result = ValidationResult(
                    name="市場數據獲取",
                    passed=True,
                    message=f"BTC/USDT: ${price} ({duration:.1f}ms)",
                )
            except Exception as e:
                result = ValidationResult(
                    name="市場數據獲取",
                    passed=False,
                    message=str(e)[:50],
                )
            self.add_result(result)

            # Check testnet mode
            result = ValidationResult(
                name="交易所模式",
                passed=True,
                message=f"{'測試網 (Testnet)' if testnet else '⚠️ 正式網 (Mainnet)'}",
            )
            self.add_result(result)

            await client.close()

        except Exception as e:
            result = ValidationResult(
                name="交易所連接",
                passed=False,
                message=str(e)[:50],
            )
            self.add_result(result)

    async def _validate_monitoring(self) -> None:
        """Validate monitoring system."""
        # Test AlertManager
        try:
            from src.monitoring import (
                AlertManager,
                AlertSeverity,
                AlertRateLimiter,
                AlertAggregator,
            )

            manager = AlertManager(
                enable_aggregation=False,
                enable_rate_limiting=True,
            )

            # Fire a test alert
            alert = await manager.fire_simple(
                severity=AlertSeverity.INFO,
                title="系統驗證測試",
                message="這是驗證測試告警",
                source="validator",
            )

            result = ValidationResult(
                name="AlertManager 功能",
                passed=alert is not None,
                message=f"Alert ID: {alert.alert_id}" if alert else "Failed to create alert",
            )
            self.add_result(result)

            # Check rate limiter
            rate_limiter = AlertRateLimiter()
            allowed = await rate_limiter.should_allow(AlertSeverity.WARNING)
            result = ValidationResult(
                name="AlertRateLimiter 功能",
                passed=allowed,
                message="速率限制正常運作",
            )
            self.add_result(result)

            # Check aggregator
            aggregator = AlertAggregator()
            result = ValidationResult(
                name="AlertAggregator 功能",
                passed=True,
                message="聚合器已初始化",
            )
            self.add_result(result)

        except Exception as e:
            result = ValidationResult(
                name="告警系統",
                passed=False,
                message=str(e)[:50],
            )
            self.add_result(result)

        # Test Watchdog
        try:
            from src.monitoring import (
                MonitoringWatchdog,
                DeadLetterQueue,
                ComponentStatus,
            )

            watchdog = MonitoringWatchdog()
            dlq = DeadLetterQueue()

            result = ValidationResult(
                name="MonitoringWatchdog 功能",
                passed=True,
                message="看門狗已初始化",
            )
            self.add_result(result)

            result = ValidationResult(
                name="DeadLetterQueue 功能",
                passed=True,
                message="死信隊列已初始化",
            )
            self.add_result(result)

        except Exception as e:
            result = ValidationResult(
                name="看門狗系統",
                passed=False,
                message=str(e)[:50],
            )
            self.add_result(result)

        # Test Comprehensive Monitor
        try:
            from src.monitoring import (
                ComprehensiveMonitor,
                WebSocketHealthMonitor,
                APIErrorMonitor,
                BalanceReconciliationMonitor,
                MarketDataStalenessMonitor,
                OrderLatencyMonitor,
            )

            monitor = ComprehensiveMonitor()

            result = ValidationResult(
                name="ComprehensiveMonitor 功能",
                passed=True,
                message="全面監控已初始化",
            )
            self.add_result(result)

            # Check sub-monitors
            monitors = [
                ("WebSocket 監控", WebSocketHealthMonitor),
                ("API 錯誤監控", APIErrorMonitor),
                ("餘額對帳監控", BalanceReconciliationMonitor),
                ("行情停滯監控", MarketDataStalenessMonitor),
                ("訂單延遲監控", OrderLatencyMonitor),
            ]

            for name, cls in monitors:
                try:
                    instance = cls()
                    result = ValidationResult(
                        name=name,
                        passed=True,
                        message="已初始化",
                    )
                except Exception as e:
                    result = ValidationResult(
                        name=name,
                        passed=False,
                        message=str(e)[:30],
                    )
                self.add_result(result)

        except Exception as e:
            result = ValidationResult(
                name="全面監控系統",
                passed=False,
                message=str(e)[:50],
            )
            self.add_result(result)

    async def _validate_alerts(self) -> None:
        """Validate alert notification channels."""
        # Test Discord webhook
        try:
            import aiohttp

            webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
            if not webhook_url:
                result = ValidationResult(
                    name="Discord Webhook",
                    passed=False,
                    message="未配置 DISCORD_WEBHOOK_URL",
                )
            else:
                # Send test message
                start = time.time()
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "embeds": [{
                            "title": "🔍 系統驗證測試",
                            "description": f"交易系統實戰資格驗證\n時間: {datetime.now(timezone.utc).isoformat()}",
                            "color": 0x3498DB,
                        }]
                    }
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        duration = (time.time() - start) * 1000
                        if resp.status in (200, 204):
                            result = ValidationResult(
                                name="Discord Webhook",
                                passed=True,
                                message=f"測試訊息已發送 ({duration:.1f}ms)",
                            )
                        else:
                            result = ValidationResult(
                                name="Discord Webhook",
                                passed=False,
                                message=f"HTTP {resp.status}",
                            )
        except Exception as e:
            result = ValidationResult(
                name="Discord Webhook",
                passed=False,
                message=str(e)[:50],
            )
        self.add_result(result)

    def _print_summary(self) -> bool:
        """Print validation summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        duration = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("📋 驗證結果摘要")
        print("=" * 70)
        print(f"\n總計檢查項目: {total}")
        print(f"✅ 通過: {passed}")
        print(f"❌ 失敗: {failed}")
        print(f"⏱️ 總耗時: {duration:.2f} 秒")

        # Failed items
        if failed > 0:
            print("\n❌ 失敗項目:")
            for r in self.results:
                if not r.passed:
                    print(f"   - {r.name}: {r.message}")

        # Production readiness verdict
        print("\n" + "=" * 70)

        # Critical checks
        critical_failures = [
            r for r in self.results
            if not r.passed and any(
                keyword in r.name.lower()
                for keyword in ["交易所", "redis", "postgresql", "api", "告警"]
            )
        ]

        if failed == 0:
            print("🎉 結論: 系統已通過所有驗證，具備實戰資格！")
            verdict = True
        elif not critical_failures:
            print("⚠️ 結論: 系統有部分非關鍵項目未通過，可以謹慎進行實戰測試")
            verdict = True
        else:
            print("🚫 結論: 系統有關鍵項目未通過，不具備實戰資格")
            print("   請先解決以下問題:")
            for r in critical_failures:
                print(f"   - {r.name}: {r.message}")
            verdict = False

        print("=" * 70 + "\n")

        return verdict


async def main():
    """Main entry point."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    validator = ProductionReadinessValidator()
    result = await validator.run_all_validations()

    return 0 if result else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
