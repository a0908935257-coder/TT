"""
Parameter Optimization Framework.

Provides various optimization algorithms for strategy parameter tuning,
including grid search, random search, and Bayesian optimization (via Optuna).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional, Union
import random
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import optuna for advanced optimization
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, GridSampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class OptimizationMethod(str, Enum):
    """Optimization method enumeration."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"  # Requires Optuna
    GENETIC = "genetic"  # Genetic Algorithm


class OptimizationDirection(str, Enum):
    """Optimization direction."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class ParameterSpace:
    """
    Definition of a parameter search space.

    Attributes:
        name: Parameter name
        param_type: Type of parameter ('int', 'float', 'categorical')
        low: Lower bound (for int/float)
        high: Upper bound (for int/float)
        step: Step size for grid search (for int/float)
        choices: List of choices (for categorical)
        log_scale: Whether to use log scale (for float)
    """

    name: str
    param_type: str = "float"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate parameter space definition."""
        if self.param_type in ("int", "float"):
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter {self.name}: low and high required for {self.param_type}")
            if self.low > self.high:
                raise ValueError(f"Parameter {self.name}: low must be <= high")
        elif self.param_type == "categorical":
            if not self.choices:
                raise ValueError(f"Parameter {self.name}: choices required for categorical")
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")

    def sample_random(self) -> Any:
        """Sample a random value from this parameter space."""
        if self.param_type == "categorical":
            return random.choice(self.choices)
        elif self.param_type == "int":
            return random.randint(int(self.low), int(self.high))
        elif self.param_type == "float":
            if self.log_scale:
                import math

                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return math.exp(random.uniform(log_low, log_high))
            return random.uniform(self.low, self.high)

    def get_grid_values(self) -> list:
        """Get all values for grid search."""
        if self.param_type == "categorical":
            return self.choices.copy()
        elif self.param_type == "int":
            step = int(self.step) if self.step else 1
            return list(range(int(self.low), int(self.high) + 1, step))
        elif self.param_type == "float":
            if self.step is None:
                # Default to 10 steps
                step = (self.high - self.low) / 10
            else:
                step = self.step
            values = []
            current = self.low
            while current <= self.high:
                values.append(round(current, 8))
                current += step
            return values


@dataclass
class OptimizationTrial:
    """
    Result of a single optimization trial.

    Attributes:
        trial_number: Trial index
        params: Parameter values used
        score: Objective function score
        metrics: Additional metrics from the trial
        duration_seconds: How long the trial took
        timestamp: When the trial was run
    """

    trial_number: int
    params: dict[str, Any]
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trial_number": self.trial_number,
            "params": self.params,
            "score": self.score,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OptimizationResult:
    """
    Complete optimization result.

    Attributes:
        best_params: Best parameter combination found
        best_score: Best score achieved
        best_metrics: Metrics from best trial
        all_trials: List of all trials
        method: Optimization method used
        direction: Optimization direction
        total_duration_seconds: Total optimization time
    """

    best_params: dict[str, Any]
    best_score: float
    best_metrics: dict[str, Any]
    all_trials: list[OptimizationTrial]
    method: OptimizationMethod
    direction: OptimizationDirection
    total_duration_seconds: float = 0.0

    @property
    def n_trials(self) -> int:
        """Get number of trials."""
        return len(self.all_trials)

    def get_top_trials(self, n: int = 10) -> list[OptimizationTrial]:
        """Get top N trials by score."""
        reverse = self.direction == OptimizationDirection.MAXIMIZE
        sorted_trials = sorted(self.all_trials, key=lambda t: t.score, reverse=reverse)
        return sorted_trials[:n]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "n_trials": self.n_trials,
            "method": self.method.value,
            "direction": self.direction.value,
            "total_duration_seconds": self.total_duration_seconds,
            "trials": [t.to_dict() for t in self.all_trials],
        }

    def summary(self) -> str:
        """Generate a summary report."""
        lines = [
            "=" * 60,
            "OPTIMIZATION RESULT SUMMARY",
            "=" * 60,
            f"Method: {self.method.value}",
            f"Direction: {self.direction.value}",
            f"Total Trials: {self.n_trials}",
            f"Duration: {self.total_duration_seconds:.2f}s",
            "",
            "BEST PARAMETERS:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"\nBest Score: {self.best_score:.6f}")

        if self.best_metrics:
            lines.append("\nBest Trial Metrics:")
            for k, v in self.best_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

        lines.append("=" * 60)
        return "\n".join(lines)


class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Subclasses implement specific optimization algorithms.
    """

    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        param_space: list[ParameterSpace],
        n_trials: int = 100,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        n_jobs: int = 1,
        callback: Optional[Callable[[OptimizationTrial], None]] = None,
    ) -> OptimizationResult:
        """
        Run optimization.

        Args:
            objective_fn: Function that takes params dict and returns (score, metrics)
            param_space: List of parameter space definitions
            n_trials: Number of trials to run
            direction: Whether to maximize or minimize
            n_jobs: Number of parallel jobs (1 = sequential)
            callback: Optional callback called after each trial

        Returns:
            OptimizationResult with best parameters and all trials
        """
        pass


class GridSearchOptimizer(Optimizer):
    """
    Grid search optimizer.

    Exhaustively searches all combinations of parameter values.
    Best for small search spaces with discrete parameters.
    """

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        param_space: list[ParameterSpace],
        n_trials: int = 100,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        n_jobs: int = 1,
        callback: Optional[Callable[[OptimizationTrial], None]] = None,
    ) -> OptimizationResult:
        """Run grid search optimization."""
        import time

        start_time = time.time()

        # Generate all combinations
        param_names = [p.name for p in param_space]
        param_values = [p.get_grid_values() for p in param_space]
        all_combinations = list(itertools.product(*param_values))

        # Limit to n_trials if specified
        if len(all_combinations) > n_trials:
            all_combinations = all_combinations[:n_trials]

        trials: list[OptimizationTrial] = []
        best_score = float("-inf") if direction == OptimizationDirection.MAXIMIZE else float("inf")
        best_params: dict[str, Any] = {}
        best_metrics: dict[str, Any] = {}

        def run_trial(idx: int, combo: tuple) -> OptimizationTrial:
            params = dict(zip(param_names, combo))
            trial_start = time.time()
            score, metrics = objective_fn(params)
            duration = time.time() - trial_start
            return OptimizationTrial(
                trial_number=idx,
                params=params,
                score=score,
                metrics=metrics,
                duration_seconds=duration,
            )

        if n_jobs == 1:
            # Sequential execution
            for idx, combo in enumerate(all_combinations):
                trial = run_trial(idx, combo)
                trials.append(trial)

                # Update best
                is_better = (
                    trial.score > best_score
                    if direction == OptimizationDirection.MAXIMIZE
                    else trial.score < best_score
                )
                if is_better:
                    best_score = trial.score
                    best_params = trial.params.copy()
                    best_metrics = trial.metrics.copy()

                if callback:
                    callback(trial)
        else:
            # Parallel execution with thread-safe best tracking
            best_lock = Lock()
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(run_trial, idx, combo): idx
                    for idx, combo in enumerate(all_combinations)
                }
                for future in as_completed(futures):
                    trial = future.result()
                    trials.append(trial)

                    is_better = (
                        trial.score > best_score
                        if direction == OptimizationDirection.MAXIMIZE
                        else trial.score < best_score
                    )
                    if is_better:
                        with best_lock:
                            # Double-check after acquiring lock
                            is_still_better = (
                                trial.score > best_score
                                if direction == OptimizationDirection.MAXIMIZE
                                else trial.score < best_score
                            )
                            if is_still_better:
                                best_score = trial.score
                                best_params = trial.params.copy()
                                best_metrics = trial.metrics.copy()

                    if callback:
                        callback(trial)

        # Sort trials by trial number
        trials.sort(key=lambda t: t.trial_number)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_trials=trials,
            method=OptimizationMethod.GRID,
            direction=direction,
            total_duration_seconds=time.time() - start_time,
        )


class RandomSearchOptimizer(Optimizer):
    """
    Random search optimizer.

    Randomly samples from the parameter space.
    More efficient than grid search for high-dimensional spaces.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize random search optimizer.

        Args:
            seed: Random seed for reproducibility
        """
        # Use instance-level RNG to avoid polluting global random state
        self._rng = random.Random(seed)

    def _sample_random(self, param: ParameterSpace) -> Any:
        """Sample a random value using instance-level RNG."""
        if param.param_type == "categorical":
            return self._rng.choice(param.choices)
        elif param.param_type == "int":
            return self._rng.randint(int(param.low), int(param.high))
        elif param.param_type == "float":
            if param.log_scale:
                import math
                log_low = math.log(param.low)
                log_high = math.log(param.high)
                return math.exp(self._rng.uniform(log_low, log_high))
            return self._rng.uniform(param.low, param.high)
        return None

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        param_space: list[ParameterSpace],
        n_trials: int = 100,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        n_jobs: int = 1,
        callback: Optional[Callable[[OptimizationTrial], None]] = None,
    ) -> OptimizationResult:
        """Run random search optimization."""
        import time

        start_time = time.time()

        trials: list[OptimizationTrial] = []
        best_score = float("-inf") if direction == OptimizationDirection.MAXIMIZE else float("inf")
        best_params: dict[str, Any] = {}
        best_metrics: dict[str, Any] = {}

        def run_trial(idx: int) -> OptimizationTrial:
            params = {p.name: self._sample_random(p) for p in param_space}
            trial_start = time.time()
            score, metrics = objective_fn(params)
            duration = time.time() - trial_start
            return OptimizationTrial(
                trial_number=idx,
                params=params,
                score=score,
                metrics=metrics,
                duration_seconds=duration,
            )

        if n_jobs == 1:
            for idx in range(n_trials):
                trial = run_trial(idx)
                trials.append(trial)

                is_better = (
                    trial.score > best_score
                    if direction == OptimizationDirection.MAXIMIZE
                    else trial.score < best_score
                )
                if is_better:
                    best_score = trial.score
                    best_params = trial.params.copy()
                    best_metrics = trial.metrics.copy()

                if callback:
                    callback(trial)
        else:
            # Parallel execution with thread-safe best tracking
            best_lock = Lock()
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(run_trial, idx): idx for idx in range(n_trials)}
                for future in as_completed(futures):
                    trial = future.result()
                    trials.append(trial)

                    is_better = (
                        trial.score > best_score
                        if direction == OptimizationDirection.MAXIMIZE
                        else trial.score < best_score
                    )
                    if is_better:
                        with best_lock:
                            # Double-check after acquiring lock
                            is_still_better = (
                                trial.score > best_score
                                if direction == OptimizationDirection.MAXIMIZE
                                else trial.score < best_score
                            )
                            if is_still_better:
                                best_score = trial.score
                                best_params = trial.params.copy()
                                best_metrics = trial.metrics.copy()

                    if callback:
                        callback(trial)

        trials.sort(key=lambda t: t.trial_number)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_trials=trials,
            method=OptimizationMethod.RANDOM,
            direction=direction,
            total_duration_seconds=time.time() - start_time,
        )


class BayesianOptimizer(Optimizer):
    """
    Bayesian optimization using Optuna's TPE sampler.

    More sample-efficient than random/grid search for expensive objectives.
    Requires Optuna to be installed.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize Bayesian optimizer.

        Args:
            seed: Random seed for reproducibility

        Raises:
            ImportError: If Optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )
        self._seed = seed

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], tuple[float, dict[str, Any]]],
        param_space: list[ParameterSpace],
        n_trials: int = 100,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        n_jobs: int = 1,
        callback: Optional[Callable[[OptimizationTrial], None]] = None,
    ) -> OptimizationResult:
        """Run Bayesian optimization using Optuna."""
        import time

        start_time = time.time()

        trials: list[OptimizationTrial] = []
        trial_metrics: dict[int, dict] = {}

        def optuna_objective(trial: "optuna.Trial") -> float:
            params = {}
            for p in param_space:
                if p.param_type == "int":
                    params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
                elif p.param_type == "float":
                    if p.log_scale:
                        params[p.name] = trial.suggest_float(p.name, p.low, p.high, log=True)
                    else:
                        params[p.name] = trial.suggest_float(p.name, p.low, p.high)
                elif p.param_type == "categorical":
                    params[p.name] = trial.suggest_categorical(p.name, p.choices)

            score, metrics = objective_fn(params)
            trial_metrics[trial.number] = metrics
            return score

        # Create Optuna study
        optuna_direction = (
            "maximize" if direction == OptimizationDirection.MAXIMIZE else "minimize"
        )
        sampler = TPESampler(seed=self._seed)
        study = optuna.create_study(direction=optuna_direction, sampler=sampler)

        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        # Convert Optuna trials to our format
        for optuna_trial in study.trials:
            trial = OptimizationTrial(
                trial_number=optuna_trial.number,
                params=optuna_trial.params,
                score=optuna_trial.value if optuna_trial.value is not None else float("nan"),
                metrics=trial_metrics.get(optuna_trial.number, {}),
                duration_seconds=optuna_trial.duration.total_seconds()
                if optuna_trial.duration
                else 0.0,
                timestamp=optuna_trial.datetime_start or datetime.now(timezone.utc),
            )
            trials.append(trial)

            if callback:
                callback(trial)

        # Get best
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        best_metrics = trial_metrics.get(best_trial.number, {})

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            all_trials=trials,
            method=OptimizationMethod.BAYESIAN,
            direction=direction,
            total_duration_seconds=time.time() - start_time,
        )


class WalkForwardOptimizer:
    """
    Walk-forward optimization with overfitting detection.

    Performs parameter optimization on in-sample data, then validates
    on out-of-sample data to detect overfitting.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        consistency_threshold: float = 0.5,
    ) -> None:
        """
        Initialize walk-forward optimizer.

        Args:
            optimizer: Base optimizer to use for each fold
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of data for training (in-sample)
            consistency_threshold: Minimum OOS/IS ratio for consistency
        """
        self._optimizer = optimizer
        self._n_splits = n_splits
        self._train_ratio = train_ratio
        self._consistency_threshold = consistency_threshold

    def optimize_with_validation(
        self,
        objective_fn: Callable[[dict[str, Any], Any], tuple[float, dict[str, Any]]],
        param_space: list[ParameterSpace],
        data_splits: list[tuple[Any, Any]],  # List of (train_data, test_data)
        n_trials: int = 50,
        direction: OptimizationDirection = OptimizationDirection.MAXIMIZE,
    ) -> dict[str, Any]:
        """
        Run walk-forward optimization.

        Args:
            objective_fn: Function(params, data) -> (score, metrics)
            param_space: Parameter space definition
            data_splits: Pre-split data as list of (train, test) tuples
            n_trials: Trials per fold
            direction: Optimization direction

        Returns:
            Dictionary with optimization results and overfitting analysis
        """
        fold_results = []
        is_scores = []
        oos_scores = []

        for fold_idx, (train_data, test_data) in enumerate(data_splits):
            # Optimize on training data
            def train_objective(params: dict) -> tuple[float, dict]:
                return objective_fn(params, train_data)

            is_result = self._optimizer.optimize(
                objective_fn=train_objective,
                param_space=param_space,
                n_trials=n_trials,
                direction=direction,
            )

            # Validate on test data
            oos_score, oos_metrics = objective_fn(is_result.best_params, test_data)

            fold_results.append(
                {
                    "fold": fold_idx,
                    "best_params": is_result.best_params,
                    "is_score": is_result.best_score,
                    "oos_score": oos_score,
                    "is_metrics": is_result.best_metrics,
                    "oos_metrics": oos_metrics,
                }
            )
            is_scores.append(is_result.best_score)
            oos_scores.append(oos_score)

        # Calculate overfitting metrics
        avg_is = sum(is_scores) / len(is_scores)
        avg_oos = sum(oos_scores) / len(oos_scores)

        if avg_is != 0:
            oos_is_ratio = avg_oos / avg_is
        else:
            oos_is_ratio = 0.0

        # Count consistent folds
        consistent_folds = sum(
            1
            for r in fold_results
            if r["is_score"] != 0 and (r["oos_score"] / r["is_score"]) >= self._consistency_threshold
        )
        consistency_pct = consistent_folds / len(fold_results) * 100

        # Detect overfitting
        is_overfitting = oos_is_ratio < self._consistency_threshold

        return {
            "fold_results": fold_results,
            "avg_is_score": avg_is,
            "avg_oos_score": avg_oos,
            "oos_is_ratio": oos_is_ratio,
            "consistency_pct": consistency_pct,
            "is_overfitting": is_overfitting,
            "n_folds": len(fold_results),
            "consistent_folds": consistent_folds,
        }


def create_optimizer(
    method: OptimizationMethod = OptimizationMethod.RANDOM,
    seed: Optional[int] = None,
    ga_config: Optional[Any] = None,
) -> Optimizer:
    """
    Factory function to create an optimizer.

    Args:
        method: Optimization method to use
        seed: Random seed for reproducibility
        ga_config: GAConfig for genetic algorithm (optional)

    Returns:
        Optimizer instance
    """
    if method == OptimizationMethod.GRID:
        return GridSearchOptimizer()
    elif method == OptimizationMethod.RANDOM:
        return RandomSearchOptimizer(seed=seed)
    elif method == OptimizationMethod.BAYESIAN:
        return BayesianOptimizer(seed=seed)
    elif method == OptimizationMethod.GENETIC:
        from .genetic_optimizer import GeneticAlgorithmOptimizer, GAConfig

        if ga_config is None:
            ga_config = GAConfig(seed=seed)
        return GeneticAlgorithmOptimizer(config=ga_config)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def is_optuna_available() -> bool:
    """Check if Optuna is available."""
    return OPTUNA_AVAILABLE
