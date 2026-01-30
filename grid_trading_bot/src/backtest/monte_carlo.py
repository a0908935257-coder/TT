"""
Monte Carlo Simulation Module.

Provides Monte Carlo methods for strategy robustness validation:
- Trade shuffling (sequence independence)
- Returns bootstrapping (statistical significance)
- Parameter perturbation (parameter sensitivity)
- Confidence interval estimation
"""

import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from .metrics import MetricsCalculator
from .result import BacktestResult, Trade


class MonteCarloMethod(str, Enum):
    """Monte Carlo simulation method."""

    TRADE_SHUFFLE = "trade_shuffle"
    RETURNS_BOOTSTRAP = "returns_bootstrap"
    PARAMETER_PERTURBATION = "parameter_perturbation"
    EQUITY_PATH = "equity_path"


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation.

    Attributes:
        n_simulations: Number of Monte Carlo iterations
        confidence_level: Confidence level for intervals (0.95 = 95%)
        seed: Random seed for reproducibility
        block_size: Block size for block bootstrap (preserves autocorrelation)
        perturbation_pct: Percentage to perturb parameters
        use_replacement: Whether to sample with replacement
    """

    n_simulations: int = 1000
    confidence_level: float = 0.95
    seed: Optional[int] = None
    block_size: int = 5
    perturbation_pct: float = 0.1
    use_replacement: bool = True


@dataclass
class ConfidenceInterval:
    """
    Confidence interval for a metric.

    Attributes:
        lower: Lower bound
        upper: Upper bound
        mean: Mean value
        median: Median value
        std: Standard deviation
        confidence_level: Confidence level (e.g., 0.95)
    """

    lower: float
    upper: float
    mean: float
    median: float
    std: float
    confidence_level: float = 0.95

    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}] (mean={self.mean:.4f}, std={self.std:.4f})"


@dataclass
class MonteCarloDistribution:
    """
    Distribution of a metric from Monte Carlo simulation.

    Attributes:
        values: All simulated values
        confidence_interval: Calculated confidence interval
        percentiles: Dictionary of percentile values
    """

    values: list[float] = field(default_factory=list)
    confidence_interval: Optional[ConfidenceInterval] = None
    percentiles: dict[int, float] = field(default_factory=dict)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0


@dataclass
class MonteCarloResult:
    """
    Complete Monte Carlo simulation result.

    Attributes:
        method: Simulation method used
        n_simulations: Number of simulations run
        original_result: Original backtest result
        total_return: Distribution of total returns
        sharpe_ratio: Distribution of Sharpe ratios
        max_drawdown: Distribution of max drawdowns
        win_rate: Distribution of win rates
        profit_factor: Distribution of profit factors
        final_equity: Distribution of final equity values
        worst_case_metrics: Metrics at worst percentile
        probability_of_profit: Probability of positive return
        probability_of_ruin: Probability of exceeding max drawdown threshold
        equity_paths: Sample of simulated equity paths
    """

    method: MonteCarloMethod
    n_simulations: int
    original_result: Optional[BacktestResult] = None

    # Metric distributions
    total_return: Optional[MonteCarloDistribution] = None
    sharpe_ratio: Optional[MonteCarloDistribution] = None
    max_drawdown: Optional[MonteCarloDistribution] = None
    win_rate: Optional[MonteCarloDistribution] = None
    profit_factor: Optional[MonteCarloDistribution] = None
    final_equity: Optional[MonteCarloDistribution] = None

    # Risk metrics
    worst_case_metrics: dict[str, float] = field(default_factory=dict)
    probability_of_profit: float = 0.0
    probability_of_ruin: float = 0.0
    ruin_threshold_pct: float = 50.0

    # Equity paths (sample)
    equity_paths: list[list[float]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "MONTE CARLO SIMULATION RESULTS",
            "=" * 60,
            f"Method: {self.method.value}",
            f"Simulations: {self.n_simulations}",
            "",
            "PROBABILITY ANALYSIS:",
            f"  Probability of Profit: {self.probability_of_profit * 100:.1f}%",
            f"  Probability of Ruin (>{self.ruin_threshold_pct}% DD): {self.probability_of_ruin * 100:.1f}%",
        ]

        if self.total_return and self.total_return.confidence_interval:
            ci = self.total_return.confidence_interval
            lines.extend([
                "",
                f"TOTAL RETURN ({ci.confidence_level * 100:.0f}% CI):",
                f"  Mean: {ci.mean:.2f}%",
                f"  Interval: [{ci.lower:.2f}%, {ci.upper:.2f}%]",
                f"  Std Dev: {ci.std:.2f}%",
            ])

        if self.sharpe_ratio and self.sharpe_ratio.confidence_interval:
            ci = self.sharpe_ratio.confidence_interval
            lines.extend([
                "",
                f"SHARPE RATIO ({ci.confidence_level * 100:.0f}% CI):",
                f"  Mean: {ci.mean:.2f}",
                f"  Interval: [{ci.lower:.2f}, {ci.upper:.2f}]",
            ])

        if self.max_drawdown and self.max_drawdown.confidence_interval:
            ci = self.max_drawdown.confidence_interval
            lines.extend([
                "",
                f"MAX DRAWDOWN ({ci.confidence_level * 100:.0f}% CI):",
                f"  Mean: {ci.mean:.2f}%",
                f"  Worst Case (95th): {self.max_drawdown.percentiles.get(95, 0):.2f}%",
            ])

        if self.worst_case_metrics:
            lines.extend([
                "",
                "WORST CASE SCENARIO (5th percentile):",
            ])
            for metric, value in self.worst_case_metrics.items():
                lines.append(f"  {metric}: {value:.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for strategy robustness analysis.

    Provides multiple simulation methods:
    1. Trade Shuffling: Randomly reorder trades to test sequence independence
    2. Returns Bootstrap: Resample returns with replacement
    3. Parameter Perturbation: Add noise to parameters and re-run backtest
    4. Equity Path: Generate synthetic equity curves

    Example:
        simulator = MonteCarloSimulator(MonteCarloConfig(n_simulations=1000))

        # Trade shuffling
        result = simulator.run_trade_shuffle(backtest_result)

        # Returns bootstrap
        result = simulator.run_returns_bootstrap(backtest_result)

        # Parameter perturbation (requires backtest function)
        result = simulator.run_parameter_perturbation(
            backtest_fn=lambda params: engine.run(klines, create_strategy(params)),
            base_params={"period": 20, "multiplier": 2.0},
            param_ranges={"period": (10, 50), "multiplier": (1.0, 4.0)},
        )

        print(result.summary())
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            config: Simulation configuration
        """
        self._config = config or MonteCarloConfig()
        if self._config.seed is not None:
            random.seed(self._config.seed)
        self._metrics_calculator = MetricsCalculator()

    @property
    def config(self) -> MonteCarloConfig:
        """Get configuration."""
        return self._config

    def run_trade_shuffle(
        self,
        result: BacktestResult,
        initial_capital: Decimal = Decimal("10000"),
    ) -> MonteCarloResult:
        """
        Run trade shuffling Monte Carlo simulation.

        Randomly reorders trades to test if strategy performance
        is independent of trade sequence. High variance indicates
        the strategy may be dependent on specific market conditions.

        Args:
            result: Original backtest result with trades
            initial_capital: Starting capital for equity calculation

        Returns:
            MonteCarloResult with distributions
        """
        if not result.trades:
            return MonteCarloResult(
                method=MonteCarloMethod.TRADE_SHUFFLE,
                n_simulations=0,
                original_result=result,
            )

        trades = result.trades
        n_sims = self._config.n_simulations

        # Storage for metrics
        total_returns: list[float] = []
        sharpe_ratios: list[float] = []
        max_drawdowns: list[float] = []
        win_rates: list[float] = []
        profit_factors: list[float] = []
        final_equities: list[float] = []
        equity_paths: list[list[float]] = []

        for sim_idx in range(n_sims):
            # Shuffle trades
            shuffled = trades.copy()
            random.shuffle(shuffled)

            # Rebuild equity curve from shuffled trades
            equity_curve = self._build_equity_from_trades(shuffled, initial_capital)

            # Calculate metrics
            metrics = self._calculate_metrics_from_shuffled(
                shuffled, equity_curve, initial_capital
            )

            total_returns.append(metrics["total_return"])
            sharpe_ratios.append(metrics["sharpe_ratio"])
            max_drawdowns.append(metrics["max_drawdown"])
            win_rates.append(metrics["win_rate"])
            profit_factors.append(metrics["profit_factor"])
            final_equities.append(float(equity_curve[-1]) if equity_curve else float(initial_capital))

            # Store sample of equity paths
            if sim_idx < 100:  # Keep first 100 paths
                equity_paths.append([float(e) for e in equity_curve])

        # Build result
        mc_result = MonteCarloResult(
            method=MonteCarloMethod.TRADE_SHUFFLE,
            n_simulations=n_sims,
            original_result=result,
            total_return=self._build_distribution(total_returns),
            sharpe_ratio=self._build_distribution(sharpe_ratios),
            max_drawdown=self._build_distribution(max_drawdowns),
            win_rate=self._build_distribution(win_rates),
            profit_factor=self._build_distribution(profit_factors),
            final_equity=self._build_distribution(final_equities),
            equity_paths=equity_paths[:10],  # Keep 10 sample paths
        )

        # Calculate risk metrics
        mc_result.probability_of_profit = sum(1 for r in total_returns if r > 0) / n_sims
        mc_result.probability_of_ruin = sum(
            1 for dd in max_drawdowns if dd > mc_result.ruin_threshold_pct
        ) / n_sims

        # Worst case metrics (5th percentile for returns, 95th for drawdown)
        mc_result.worst_case_metrics = {
            "total_return": self._percentile(total_returns, 5),
            "sharpe_ratio": self._percentile(sharpe_ratios, 5),
            "max_drawdown": self._percentile(max_drawdowns, 95),
            "profit_factor": self._percentile(profit_factors, 5),
        }

        return mc_result

    def run_returns_bootstrap(
        self,
        result: BacktestResult,
        initial_capital: Decimal = Decimal("10000"),
    ) -> MonteCarloResult:
        """
        Run returns bootstrapping Monte Carlo simulation.

        Resamples daily/trade returns with replacement to estimate
        the statistical significance of observed performance.
        Uses block bootstrap to preserve some autocorrelation.

        Args:
            result: Original backtest result
            initial_capital: Starting capital

        Returns:
            MonteCarloResult with distributions
        """
        # Get returns from trades
        if not result.trades:
            return MonteCarloResult(
                method=MonteCarloMethod.RETURNS_BOOTSTRAP,
                n_simulations=0,
                original_result=result,
            )

        returns = [float(t.pnl_pct) for t in result.trades]
        n_returns = len(returns)
        n_sims = self._config.n_simulations
        block_size = min(self._config.block_size, n_returns)

        # Storage
        total_returns: list[float] = []
        sharpe_ratios: list[float] = []
        max_drawdowns: list[float] = []
        final_equities: list[float] = []
        equity_paths: list[list[float]] = []

        for sim_idx in range(n_sims):
            # Block bootstrap
            bootstrapped_returns = self._block_bootstrap(returns, n_returns, block_size)

            # Build equity curve
            equity = float(initial_capital)
            equity_curve = [equity]

            for ret_pct in bootstrapped_returns:
                equity *= (1 + ret_pct / 100)
                # Clamp to prevent float overflow on high leverage
                if equity > 1e15:
                    equity = 1e15
                elif equity < 0:
                    equity = 0
                equity_curve.append(equity)

            # Calculate metrics
            total_ret = (equity_curve[-1] / float(initial_capital) - 1) * 100
            total_returns.append(total_ret)

            # Simple Sharpe calculation
            if len(bootstrapped_returns) > 1:
                mean_ret = math.fsum(bootstrapped_returns) / len(bootstrapped_returns)
                try:
                    std_ret = math.sqrt(
                        math.fsum((r - mean_ret) ** 2 for r in bootstrapped_returns)
                        / (len(bootstrapped_returns) - 1)
                    )
                except (OverflowError, ValueError):
                    std_ret = float('inf')
                sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 and math.isfinite(std_ret) else 0
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

            # Max drawdown
            max_dd = self._calculate_drawdown(equity_curve)
            max_drawdowns.append(max_dd)

            final_equities.append(equity_curve[-1])

            if sim_idx < 100:
                equity_paths.append(equity_curve)

        # Build result
        mc_result = MonteCarloResult(
            method=MonteCarloMethod.RETURNS_BOOTSTRAP,
            n_simulations=n_sims,
            original_result=result,
            total_return=self._build_distribution(total_returns),
            sharpe_ratio=self._build_distribution(sharpe_ratios),
            max_drawdown=self._build_distribution(max_drawdowns),
            final_equity=self._build_distribution(final_equities),
            equity_paths=equity_paths[:10],
        )

        mc_result.probability_of_profit = sum(1 for r in total_returns if r > 0) / n_sims
        mc_result.probability_of_ruin = sum(
            1 for dd in max_drawdowns if dd > mc_result.ruin_threshold_pct
        ) / n_sims

        mc_result.worst_case_metrics = {
            "total_return": self._percentile(total_returns, 5),
            "sharpe_ratio": self._percentile(sharpe_ratios, 5),
            "max_drawdown": self._percentile(max_drawdowns, 95),
        }

        return mc_result

    def run_parameter_perturbation(
        self,
        backtest_fn: Callable[[dict[str, Any]], BacktestResult],
        base_params: dict[str, Any],
        param_ranges: Optional[dict[str, tuple[float, float]]] = None,
    ) -> MonteCarloResult:
        """
        Run parameter perturbation Monte Carlo simulation.

        Adds random noise to parameters and re-runs backtest to test
        parameter sensitivity. High variance indicates the strategy
        is sensitive to parameter choices (potential overfitting).

        Args:
            backtest_fn: Function that takes params and returns BacktestResult
            base_params: Base parameter values
            param_ranges: Optional ranges for each parameter (min, max)
                         If not provided, uses +/- perturbation_pct

        Returns:
            MonteCarloResult with distributions
        """
        n_sims = self._config.n_simulations
        perturbation_pct = self._config.perturbation_pct

        # Storage
        total_returns: list[float] = []
        sharpe_ratios: list[float] = []
        max_drawdowns: list[float] = []
        win_rates: list[float] = []
        profit_factors: list[float] = []
        param_samples: list[dict[str, Any]] = []

        # Run original for reference
        original_result = backtest_fn(base_params)

        for sim_idx in range(n_sims):
            # Perturb parameters
            perturbed_params = {}
            for name, value in base_params.items():
                if isinstance(value, (int, float)):
                    if param_ranges and name in param_ranges:
                        min_val, max_val = param_ranges[name]
                    else:
                        min_val = value * (1 - perturbation_pct)
                        max_val = value * (1 + perturbation_pct)

                    # Sample from range
                    new_value = random.uniform(min_val, max_val)

                    # Preserve int type
                    if isinstance(value, int):
                        new_value = int(round(new_value))

                    perturbed_params[name] = new_value
                else:
                    perturbed_params[name] = value

            # Run backtest
            try:
                result = backtest_fn(perturbed_params)

                total_returns.append(float(result.total_profit_pct))
                sharpe_ratios.append(float(result.sharpe_ratio))
                max_drawdowns.append(float(result.max_drawdown_pct))
                win_rates.append(float(result.win_rate))
                profit_factors.append(float(result.profit_factor))

                if sim_idx < 50:
                    param_samples.append(perturbed_params.copy())

            except Exception:
                # Skip failed backtests
                continue

        if not total_returns:
            return MonteCarloResult(
                method=MonteCarloMethod.PARAMETER_PERTURBATION,
                n_simulations=0,
                original_result=original_result,
            )

        # Build result
        actual_sims = len(total_returns)
        mc_result = MonteCarloResult(
            method=MonteCarloMethod.PARAMETER_PERTURBATION,
            n_simulations=actual_sims,
            original_result=original_result,
            total_return=self._build_distribution(total_returns),
            sharpe_ratio=self._build_distribution(sharpe_ratios),
            max_drawdown=self._build_distribution(max_drawdowns),
            win_rate=self._build_distribution(win_rates),
            profit_factor=self._build_distribution(profit_factors),
        )

        mc_result.probability_of_profit = sum(1 for r in total_returns if r > 0) / actual_sims
        mc_result.probability_of_ruin = sum(
            1 for dd in max_drawdowns if dd > mc_result.ruin_threshold_pct
        ) / actual_sims

        mc_result.worst_case_metrics = {
            "total_return": self._percentile(total_returns, 5),
            "sharpe_ratio": self._percentile(sharpe_ratios, 5),
            "max_drawdown": self._percentile(max_drawdowns, 95),
            "win_rate": self._percentile(win_rates, 5),
            "profit_factor": self._percentile(profit_factors, 5),
        }

        return mc_result

    def run_equity_path_simulation(
        self,
        result: BacktestResult,
        initial_capital: Decimal = Decimal("10000"),
        periods: int = 252,
    ) -> MonteCarloResult:
        """
        Generate synthetic equity paths using GBM (Geometric Brownian Motion).

        Uses observed mean and volatility of returns to simulate
        future equity paths. Useful for risk projection.

        Args:
            result: Original backtest result
            initial_capital: Starting capital
            periods: Number of periods to simulate

        Returns:
            MonteCarloResult with simulated paths
        """
        if not result.trades:
            return MonteCarloResult(
                method=MonteCarloMethod.EQUITY_PATH,
                n_simulations=0,
                original_result=result,
            )

        # Calculate return statistics
        returns = [float(t.pnl_pct) / 100 for t in result.trades]  # Convert to decimal
        mu = sum(returns) / len(returns)  # Mean return
        sigma = math.sqrt(sum((r - mu) ** 2 for r in returns) / max(len(returns) - 1, 1))

        n_sims = self._config.n_simulations

        # Storage
        final_equities: list[float] = []
        max_drawdowns: list[float] = []
        total_returns: list[float] = []
        equity_paths: list[list[float]] = []

        for sim_idx in range(n_sims):
            # Simulate GBM path
            equity = float(initial_capital)
            path = [equity]
            peak = equity
            max_dd = 0.0

            for _ in range(periods):
                # Random return from normal distribution
                random_return = random.gauss(mu, sigma)
                # Limit loss to -100% to prevent negative equity
                random_return = max(random_return, -1.0)
                equity *= (1 + random_return)
                # Ensure equity doesn't go negative or overflow on high leverage
                equity = max(equity, 0.0)
                if equity > 1e15:
                    equity = 1e15
                path.append(equity)

                # Track drawdown
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            total_returns.append((equity / float(initial_capital) - 1) * 100)

            if sim_idx < 100:
                equity_paths.append(path)

        # Build result
        mc_result = MonteCarloResult(
            method=MonteCarloMethod.EQUITY_PATH,
            n_simulations=n_sims,
            original_result=result,
            total_return=self._build_distribution(total_returns),
            max_drawdown=self._build_distribution(max_drawdowns),
            final_equity=self._build_distribution(final_equities),
            equity_paths=equity_paths[:10],
        )

        mc_result.probability_of_profit = sum(1 for r in total_returns if r > 0) / n_sims
        mc_result.probability_of_ruin = sum(
            1 for dd in max_drawdowns if dd > mc_result.ruin_threshold_pct
        ) / n_sims

        mc_result.worst_case_metrics = {
            "total_return": self._percentile(total_returns, 5),
            "max_drawdown": self._percentile(max_drawdowns, 95),
            "final_equity": self._percentile(final_equities, 5),
        }

        return mc_result

    def _build_equity_from_trades(
        self,
        trades: list[Trade],
        initial_capital: Decimal,
    ) -> list[Decimal]:
        """Build equity curve from trade sequence."""
        equity = initial_capital
        curve = [equity]

        for trade in trades:
            equity += trade.pnl
            curve.append(equity)

        return curve

    def _calculate_metrics_from_shuffled(
        self,
        trades: list[Trade],
        equity_curve: list[Decimal],
        initial_capital: Decimal,
    ) -> dict[str, float]:
        """Calculate metrics from shuffled trades."""
        if not trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        total_pnl = sum(t.pnl for t in trades)
        total_return = float(total_pnl / initial_capital * 100)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades) * 100 if trades else 0

        gross_profit = sum(t.pnl for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else Decimal("0")
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 999.0

        # Max drawdown
        max_dd = self._calculate_drawdown([float(e) for e in equity_curve])

        # Simple Sharpe from trade returns
        if len(trades) > 1:
            returns = [float(t.pnl_pct) for t in trades]
            mean_ret = sum(returns) / len(returns)
            std_ret = math.sqrt(
                sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            )
            sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0
        else:
            sharpe = 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def _calculate_drawdown(self, equity_curve: list[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _block_bootstrap(
        self,
        data: list[float],
        n_samples: int,
        block_size: int,
    ) -> list[float]:
        """Perform block bootstrap sampling."""
        if not data:
            return []

        result = []
        n = len(data)

        # Ensure block_size doesn't exceed data length to avoid excessive wrapping
        effective_block_size = min(block_size, n)

        while len(result) < n_samples:
            # Random start position
            start = random.randint(0, n - 1)

            # Extract block (wrap around if needed)
            for i in range(effective_block_size):
                if len(result) >= n_samples:
                    break
                idx = (start + i) % n
                result.append(data[idx])

        return result[:n_samples]

    def _build_distribution(self, values: list[float]) -> MonteCarloDistribution:
        """Build distribution with confidence interval."""
        if not values:
            return MonteCarloDistribution()

        sorted_values = sorted(values)
        n = len(sorted_values)

        # Calculate statistics (safe for extreme leverage values)
        mean = math.fsum(values) / n
        median = sorted_values[n // 2]
        try:
            variance = math.fsum((x - mean) ** 2 for x in values) / max(n - 1, 1)
            std = math.sqrt(variance)
        except (OverflowError, ValueError):
            from decimal import Decimal as D
            d_mean = D(str(mean))
            variance = float(sum((D(str(x)) - d_mean) ** 2 for x in values) / max(n - 1, 1))
            std = math.sqrt(abs(variance))

        # Confidence interval
        alpha = 1 - self._config.confidence_level
        lower_idx = int(n * alpha / 2)
        upper_idx = int(n * (1 - alpha / 2))

        ci = ConfidenceInterval(
            lower=sorted_values[lower_idx],
            upper=sorted_values[min(upper_idx, n - 1)],
            mean=mean,
            median=median,
            std=std,
            confidence_level=self._config.confidence_level,
        )

        # Percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            idx = int(n * p / 100)
            percentiles[p] = sorted_values[min(idx, n - 1)]

        return MonteCarloDistribution(
            values=values,
            confidence_interval=ci,
            percentiles=percentiles,
        )

    def _percentile(self, values: list[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]


def run_monte_carlo_validation(
    result: BacktestResult,
    methods: Optional[list[MonteCarloMethod]] = None,
    config: Optional[MonteCarloConfig] = None,
    initial_capital: Decimal = Decimal("10000"),
) -> dict[MonteCarloMethod, MonteCarloResult]:
    """
    Run multiple Monte Carlo validation methods.

    Convenience function to run multiple validation methods at once.

    Args:
        result: Backtest result to validate
        methods: List of methods to run (default: all except parameter perturbation)
        config: Monte Carlo configuration
        initial_capital: Starting capital

    Returns:
        Dictionary of method to result
    """
    if methods is None:
        methods = [
            MonteCarloMethod.TRADE_SHUFFLE,
            MonteCarloMethod.RETURNS_BOOTSTRAP,
            MonteCarloMethod.EQUITY_PATH,
        ]

    simulator = MonteCarloSimulator(config)
    results = {}

    for method in methods:
        if method == MonteCarloMethod.TRADE_SHUFFLE:
            results[method] = simulator.run_trade_shuffle(result, initial_capital)
        elif method == MonteCarloMethod.RETURNS_BOOTSTRAP:
            results[method] = simulator.run_returns_bootstrap(result, initial_capital)
        elif method == MonteCarloMethod.EQUITY_PATH:
            results[method] = simulator.run_equity_path_simulation(result, initial_capital)

    return results


def generate_monte_carlo_report(
    results: dict[MonteCarloMethod, MonteCarloResult],
    format: str = "text",
) -> str:
    """
    Generate a comprehensive Monte Carlo validation report.

    Args:
        results: Dictionary of Monte Carlo results
        format: Output format ('text' or 'markdown')

    Returns:
        Formatted report string
    """
    if format == "markdown":
        return _generate_markdown_report(results)
    return _generate_text_report(results)


def _generate_text_report(results: dict[MonteCarloMethod, MonteCarloResult]) -> str:
    """Generate text format report."""
    lines = [
        "=" * 70,
        "MONTE CARLO VALIDATION REPORT",
        "=" * 70,
        "",
    ]

    for method, result in results.items():
        lines.extend([
            f"[{method.value.upper()}]",
            f"Simulations: {result.n_simulations}",
            f"Probability of Profit: {result.probability_of_profit * 100:.1f}%",
            f"Probability of Ruin: {result.probability_of_ruin * 100:.1f}%",
            "",
        ])

        if result.total_return and result.total_return.confidence_interval:
            ci = result.total_return.confidence_interval
            lines.append(
                f"Total Return: {ci.mean:.2f}% [{ci.lower:.2f}%, {ci.upper:.2f}%]"
            )

        if result.sharpe_ratio and result.sharpe_ratio.confidence_interval:
            ci = result.sharpe_ratio.confidence_interval
            lines.append(f"Sharpe Ratio: {ci.mean:.2f} [{ci.lower:.2f}, {ci.upper:.2f}]")

        if result.max_drawdown and result.max_drawdown.confidence_interval:
            ci = result.max_drawdown.confidence_interval
            lines.append(
                f"Max Drawdown: {ci.mean:.2f}% (95th: {result.max_drawdown.percentiles.get(95, 0):.2f}%)"
            )

        lines.extend(["", "-" * 70, ""])

    # Summary
    if results:
        first_result = list(results.values())[0]
        if first_result.original_result:
            orig = first_result.original_result
            lines.extend([
                "ORIGINAL BACKTEST COMPARISON:",
                f"  Total Return: {orig.total_profit_pct:.2f}%",
                f"  Sharpe Ratio: {orig.sharpe_ratio:.2f}",
                f"  Max Drawdown: {orig.max_drawdown_pct:.2f}%",
            ])

    lines.append("=" * 70)
    return "\n".join(lines)


def _generate_markdown_report(results: dict[MonteCarloMethod, MonteCarloResult]) -> str:
    """Generate markdown format report."""
    lines = [
        "# Monte Carlo Validation Report",
        "",
    ]

    for method, result in results.items():
        lines.extend([
            f"## {method.value.replace('_', ' ').title()}",
            "",
            f"- **Simulations**: {result.n_simulations}",
            f"- **Probability of Profit**: {result.probability_of_profit * 100:.1f}%",
            f"- **Probability of Ruin**: {result.probability_of_ruin * 100:.1f}%",
            "",
        ])

        if result.total_return and result.total_return.confidence_interval:
            ci = result.total_return.confidence_interval
            lines.extend([
                "### Total Return Distribution",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean | {ci.mean:.2f}% |",
                f"| 95% CI | [{ci.lower:.2f}%, {ci.upper:.2f}%] |",
                f"| Std Dev | {ci.std:.2f}% |",
                "",
            ])

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
