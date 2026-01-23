"""
Unit tests for parameter optimization framework.
"""

import pytest
from decimal import Decimal

from src.backtest.optimizer import (
    ParameterSpace,
    OptimizationMethod,
    OptimizationDirection,
    OptimizationTrial,
    OptimizationResult,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    WalkForwardOptimizer,
    create_optimizer,
    is_optuna_available,
)


class TestParameterSpace:
    """Tests for ParameterSpace class."""

    def test_int_parameter(self):
        """Test integer parameter space."""
        space = ParameterSpace(
            name="grid_count",
            param_type="int",
            low=5,
            high=20,
            step=5,
        )
        values = space.get_grid_values()
        assert values == [5, 10, 15, 20]

    def test_float_parameter(self):
        """Test float parameter space."""
        space = ParameterSpace(
            name="threshold",
            param_type="float",
            low=0.0,
            high=1.0,
            step=0.25,
        )
        values = space.get_grid_values()
        assert len(values) == 5  # 0.0, 0.25, 0.5, 0.75, 1.0

    def test_categorical_parameter(self):
        """Test categorical parameter space."""
        space = ParameterSpace(
            name="mode",
            param_type="categorical",
            choices=["aggressive", "conservative", "neutral"],
        )
        values = space.get_grid_values()
        assert values == ["aggressive", "conservative", "neutral"]

    def test_random_sampling_int(self):
        """Test random sampling for int parameter."""
        space = ParameterSpace(
            name="count",
            param_type="int",
            low=1,
            high=100,
        )
        for _ in range(10):
            value = space.sample_random()
            assert isinstance(value, int)
            assert 1 <= value <= 100

    def test_random_sampling_float(self):
        """Test random sampling for float parameter."""
        space = ParameterSpace(
            name="rate",
            param_type="float",
            low=0.0,
            high=1.0,
        )
        for _ in range(10):
            value = space.sample_random()
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

    def test_random_sampling_categorical(self):
        """Test random sampling for categorical parameter."""
        choices = ["a", "b", "c"]
        space = ParameterSpace(
            name="choice",
            param_type="categorical",
            choices=choices,
        )
        for _ in range(10):
            value = space.sample_random()
            assert value in choices

    def test_invalid_int_missing_bounds(self):
        """Test int parameter without bounds raises error."""
        with pytest.raises(ValueError, match="low and high required"):
            ParameterSpace(name="x", param_type="int")

    def test_invalid_categorical_no_choices(self):
        """Test categorical without choices raises error."""
        with pytest.raises(ValueError, match="choices required"):
            ParameterSpace(name="x", param_type="categorical")

    def test_invalid_param_type(self):
        """Test unknown param_type raises error."""
        with pytest.raises(ValueError, match="Unknown param_type"):
            ParameterSpace(name="x", param_type="unknown")


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    def test_basic_optimization(self):
        """Test basic grid search optimization."""

        def objective(params):
            # Simple quadratic function with maximum at x=5
            x = params["x"]
            score = -(x - 5) ** 2
            return score, {"x": x}

        optimizer = GridSearchOptimizer()
        param_space = [
            ParameterSpace(name="x", param_type="int", low=0, high=10, step=1)
        ]

        result = optimizer.optimize(
            objective_fn=objective,
            param_space=param_space,
            n_trials=100,
            direction=OptimizationDirection.MAXIMIZE,
        )

        assert result.best_params["x"] == 5
        assert result.best_score == 0
        assert result.method == OptimizationMethod.GRID
        assert len(result.all_trials) == 11  # 0 to 10 inclusive

    def test_minimization(self):
        """Test grid search minimization."""

        def objective(params):
            x = params["x"]
            score = (x - 3) ** 2  # Minimum at x=3
            return score, {}

        optimizer = GridSearchOptimizer()
        param_space = [
            ParameterSpace(name="x", param_type="int", low=0, high=10, step=1)
        ]

        result = optimizer.optimize(
            objective_fn=objective,
            param_space=param_space,
            direction=OptimizationDirection.MINIMIZE,
        )

        assert result.best_params["x"] == 3
        assert result.best_score == 0

    def test_multi_parameter(self):
        """Test grid search with multiple parameters."""

        def objective(params):
            x = params["x"]
            y = params["y"]
            score = -(x - 2) ** 2 - (y - 3) ** 2
            return score, {"x": x, "y": y}

        optimizer = GridSearchOptimizer()
        param_space = [
            ParameterSpace(name="x", param_type="int", low=0, high=4, step=1),
            ParameterSpace(name="y", param_type="int", low=0, high=6, step=1),
        ]

        result = optimizer.optimize(
            objective_fn=objective,
            param_space=param_space,
            direction=OptimizationDirection.MAXIMIZE,
        )

        assert result.best_params["x"] == 2
        assert result.best_params["y"] == 3

    def test_callback(self):
        """Test callback is called for each trial."""
        callback_count = [0]

        def callback(trial):
            callback_count[0] += 1

        def objective(params):
            return params["x"], {}

        optimizer = GridSearchOptimizer()
        param_space = [
            ParameterSpace(name="x", param_type="int", low=1, high=5, step=1)
        ]

        result = optimizer.optimize(
            objective_fn=objective,
            param_space=param_space,
            callback=callback,
        )

        assert callback_count[0] == 5


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""

    def test_basic_optimization(self):
        """Test basic random search."""

        def objective(params):
            x = params["x"]
            score = -(x - 5) ** 2
            return score, {}

        optimizer = RandomSearchOptimizer(seed=42)
        param_space = [
            ParameterSpace(name="x", param_type="int", low=0, high=10)
        ]

        result = optimizer.optimize(
            objective_fn=objective,
            param_space=param_space,
            n_trials=50,
            direction=OptimizationDirection.MAXIMIZE,
        )

        # Should find something close to 5
        assert abs(result.best_params["x"] - 5) <= 2
        assert result.method == OptimizationMethod.RANDOM
        assert len(result.all_trials) == 50

    def test_reproducibility(self):
        """Test random search is reproducible with seed."""

        def objective(params):
            return params["x"], {}

        param_space = [
            ParameterSpace(name="x", param_type="float", low=0.0, high=1.0)
        ]

        optimizer1 = RandomSearchOptimizer(seed=123)
        result1 = optimizer1.optimize(
            objective_fn=objective,
            param_space=param_space,
            n_trials=10,
        )

        optimizer2 = RandomSearchOptimizer(seed=123)
        result2 = optimizer2.optimize(
            objective_fn=objective,
            param_space=param_space,
            n_trials=10,
        )

        # Results should be identical with same seed
        for t1, t2 in zip(result1.all_trials, result2.all_trials):
            assert t1.params["x"] == t2.params["x"]


class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_get_top_trials(self):
        """Test getting top N trials."""
        trials = [
            OptimizationTrial(i, {"x": i}, float(i), {})
            for i in range(10)
        ]
        result = OptimizationResult(
            best_params={"x": 9},
            best_score=9.0,
            best_metrics={},
            all_trials=trials,
            method=OptimizationMethod.RANDOM,
            direction=OptimizationDirection.MAXIMIZE,
        )

        top_3 = result.get_top_trials(3)
        assert len(top_3) == 3
        assert top_3[0].score == 9.0
        assert top_3[1].score == 8.0
        assert top_3[2].score == 7.0

    def test_summary(self):
        """Test summary generation."""
        result = OptimizationResult(
            best_params={"x": 5, "y": 3},
            best_score=0.95,
            best_metrics={"accuracy": 0.95, "loss": 0.05},
            all_trials=[],
            method=OptimizationMethod.GRID,
            direction=OptimizationDirection.MAXIMIZE,
            total_duration_seconds=10.5,
        )

        summary = result.summary()
        assert "OPTIMIZATION RESULT" in summary
        assert "x: 5" in summary
        assert "y: 3" in summary
        assert "0.95" in summary

    def test_to_dict(self):
        """Test serialization to dict."""
        result = OptimizationResult(
            best_params={"x": 5},
            best_score=0.95,
            best_metrics={"acc": 0.95},
            all_trials=[
                OptimizationTrial(0, {"x": 5}, 0.95, {"acc": 0.95})
            ],
            method=OptimizationMethod.GRID,
            direction=OptimizationDirection.MAXIMIZE,
        )

        d = result.to_dict()
        assert d["best_params"] == {"x": 5}
        assert d["best_score"] == 0.95
        assert d["method"] == "grid"
        assert len(d["trials"]) == 1


class TestCreateOptimizer:
    """Tests for create_optimizer factory function."""

    def test_create_grid(self):
        """Test creating grid search optimizer."""
        optimizer = create_optimizer(OptimizationMethod.GRID)
        assert isinstance(optimizer, GridSearchOptimizer)

    def test_create_random(self):
        """Test creating random search optimizer."""
        optimizer = create_optimizer(OptimizationMethod.RANDOM, seed=42)
        assert isinstance(optimizer, RandomSearchOptimizer)

    def test_create_bayesian_without_optuna(self):
        """Test Bayesian optimizer without Optuna raises error."""
        if not is_optuna_available():
            with pytest.raises(ImportError, match="Optuna is required"):
                create_optimizer(OptimizationMethod.BAYESIAN)


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    def test_basic_walk_forward(self):
        """Test basic walk-forward optimization."""

        def objective(params, data):
            # Simple objective: return sum of data * param
            x = params["x"]
            score = sum(d * x for d in data)
            return score, {"data_len": len(data)}

        base_optimizer = RandomSearchOptimizer(seed=42)
        wf_optimizer = WalkForwardOptimizer(
            optimizer=base_optimizer,
            n_splits=3,
            consistency_threshold=0.3,
        )

        param_space = [
            ParameterSpace(name="x", param_type="float", low=0.1, high=2.0)
        ]

        # Create data splits
        data_splits = [
            ([1, 2, 3], [4, 5]),      # Fold 1
            ([2, 3, 4], [5, 6]),      # Fold 2
            ([3, 4, 5], [6, 7]),      # Fold 3
        ]

        result = wf_optimizer.optimize_with_validation(
            objective_fn=objective,
            param_space=param_space,
            data_splits=data_splits,
            n_trials=10,
            direction=OptimizationDirection.MAXIMIZE,
        )

        assert "fold_results" in result
        assert result["n_folds"] == 3
        assert "avg_is_score" in result
        assert "avg_oos_score" in result
        assert "oos_is_ratio" in result
        assert "is_overfitting" in result
        assert "consistency_pct" in result
