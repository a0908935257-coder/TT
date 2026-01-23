"""
Unit tests for experiment tracking module.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from src.backtest.experiment import (
    ExperimentTracker,
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig class."""

    def test_default_config(self):
        """Test default config creation."""
        config = ExperimentConfig(
            strategy_name="grid_strategy",
            strategy_version="1.0.0",
            strategy_params={"grid_count": 10},
            backtest_config={"initial_capital": 10000},
            data_source="binance",
            data_range=("2024-01-01", "2024-06-01"),
        )
        assert config.strategy_name == "grid_strategy"
        assert config.strategy_version == "1.0.0"
        assert config.strategy_params["grid_count"] == 10

    def test_to_dict(self):
        """Test serialization to dict."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={"x": 1},
            backtest_config={},
            data_source="local",
            data_range=("2024-01-01", "2024-12-31"),
        )
        d = config.to_dict()
        assert d["strategy_name"] == "test"
        assert d["strategy_version"] == "1.0.0"
        assert d["data_source"] == "local"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "strategy_name": "test",
            "strategy_version": "2.0.0",
            "strategy_params": {"a": 1, "b": 2},
            "backtest_config": {"capital": 5000},
            "data_source": "external",
            "data_range": ["2024-01-01", "2024-03-01"],
        }
        config = ExperimentConfig.from_dict(data)
        assert config.strategy_name == "test"
        assert config.strategy_version == "2.0.0"
        assert config.data_range == ("2024-01-01", "2024-03-01")


class TestExperimentResult:
    """Tests for ExperimentResult class."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = ExperimentResult(
            metrics={"sharpe": 1.5, "max_dd": 0.15},
            trades_count=100,
        )
        assert result.metrics["sharpe"] == 1.5
        assert result.trades_count == 100
        assert result.equity_curve is None

    def test_result_with_equity_curve(self):
        """Test result with equity curve."""
        result = ExperimentResult(
            metrics={"return": 0.25},
            trades_count=50,
            equity_curve=[10000, 10100, 10250, 10200, 10500],
        )
        assert len(result.equity_curve) == 5
        assert result.equity_curve[-1] == 10500

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ExperimentResult(
            metrics={"sharpe": 2.0},
            trades_count=75,
            walk_forward_results={"is_overfitting": False},
        )
        d = result.to_dict()
        assert d["metrics"]["sharpe"] == 2.0
        assert d["trades_count"] == 75
        assert d["walk_forward_results"]["is_overfitting"] is False

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "metrics": {"return": 0.3, "sharpe": 1.8},
            "trades_count": 120,
            "equity_curve": [100, 110, 105, 115],
            "walk_forward_results": None,
            "optimization_results": {"best_param": 5},
        }
        result = ExperimentResult.from_dict(data)
        assert result.metrics["return"] == 0.3
        assert result.trades_count == 120
        assert len(result.equity_curve) == 4


class TestExperiment:
    """Tests for Experiment class."""

    def test_experiment_creation(self):
        """Test experiment creation."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = Experiment(
            experiment_id="exp_001",
            name="Test Experiment",
            config=config,
        )
        assert exp.experiment_id == "exp_001"
        assert exp.name == "Test Experiment"
        assert exp.status == ExperimentStatus.PENDING

    def test_experiment_with_tags(self):
        """Test experiment with tags."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = Experiment(
            experiment_id="exp_002",
            name="Tagged Experiment",
            config=config,
            tags=["production", "v2"],
        )
        assert "production" in exp.tags
        assert "v2" in exp.tags

    def test_to_dict_and_from_dict(self):
        """Test roundtrip serialization."""
        config = ExperimentConfig(
            strategy_name="grid",
            strategy_version="1.0.0",
            strategy_params={"count": 10},
            backtest_config={"capital": 10000},
            data_source="binance",
            data_range=("2024-01-01", "2024-06-01"),
        )
        original = Experiment(
            experiment_id="exp_003",
            name="Roundtrip Test",
            config=config,
            tags=["test"],
        )
        d = original.to_dict()
        restored = Experiment.from_dict(d)

        assert restored.experiment_id == original.experiment_id
        assert restored.name == original.name
        assert restored.config.strategy_name == original.config.strategy_name
        assert restored.tags == original.tags


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def tracker(self, temp_dir):
        """Create a tracker with temp directory."""
        return ExperimentTracker(base_path=temp_dir)

    def test_create_experiment(self, tracker):
        """Test creating an experiment."""
        config = ExperimentConfig(
            strategy_name="grid",
            strategy_version="1.0.0",
            strategy_params={"grid_count": 10},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-03-01"),
        )
        exp = tracker.create_experiment(
            name="My Experiment",
            config=config,
            tags=["test"],
        )

        assert exp.experiment_id is not None
        assert exp.name == "My Experiment"
        assert exp.status == ExperimentStatus.PENDING
        assert "test" in exp.tags

    def test_start_experiment(self, tracker):
        """Test starting an experiment."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = tracker.create_experiment("Test", config)
        started = tracker.start_experiment(exp.experiment_id)

        assert started.status == ExperimentStatus.RUNNING
        assert started.started_at is not None

    def test_complete_experiment(self, tracker):
        """Test completing an experiment."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = tracker.create_experiment("Test", config)
        tracker.start_experiment(exp.experiment_id)

        result = ExperimentResult(
            metrics={"sharpe": 1.5, "return": 0.25},
            trades_count=50,
        )
        completed = tracker.complete_experiment(exp.experiment_id, result)

        assert completed.status == ExperimentStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.result.metrics["sharpe"] == 1.5

    def test_fail_experiment(self, tracker):
        """Test failing an experiment."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = tracker.create_experiment("Test", config)
        tracker.start_experiment(exp.experiment_id)
        failed = tracker.fail_experiment(exp.experiment_id, "Test error message")

        assert failed.status == ExperimentStatus.FAILED
        assert failed.result.error_message == "Test error message"

    def test_get_experiment(self, tracker):
        """Test getting an experiment by ID."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = tracker.create_experiment("Test", config)
        retrieved = tracker.get_experiment(exp.experiment_id)

        assert retrieved is not None
        assert retrieved.experiment_id == exp.experiment_id
        assert retrieved.name == exp.name

    def test_get_nonexistent_experiment(self, tracker):
        """Test getting a nonexistent experiment returns None."""
        result = tracker.get_experiment("nonexistent_id")
        assert result is None

    def test_list_experiments(self, tracker):
        """Test listing experiments."""
        config = ExperimentConfig(
            strategy_name="grid",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        # Create multiple experiments
        for i in range(3):
            tracker.create_experiment(f"Experiment {i}", config)

        experiments = tracker.list_experiments()
        assert len(experiments) == 3

    def test_list_experiments_by_strategy(self, tracker):
        """Test listing experiments filtered by strategy."""
        config1 = ExperimentConfig(
            strategy_name="grid",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        config2 = ExperimentConfig(
            strategy_name="supertrend",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        tracker.create_experiment("Grid 1", config1)
        tracker.create_experiment("Grid 2", config1)
        tracker.create_experiment("Supertrend 1", config2)

        grid_experiments = tracker.list_experiments(strategy_name="grid")
        assert len(grid_experiments) == 2

    def test_list_experiments_by_status(self, tracker):
        """Test listing experiments filtered by status."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        exp1 = tracker.create_experiment("Pending", config)
        exp2 = tracker.create_experiment("Running", config)
        tracker.start_experiment(exp2.experiment_id)

        pending = tracker.list_experiments(status=ExperimentStatus.PENDING)
        running = tracker.list_experiments(status=ExperimentStatus.RUNNING)

        assert len(pending) == 1
        assert len(running) == 1

    def test_list_experiments_by_tags(self, tracker):
        """Test listing experiments filtered by tags."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        tracker.create_experiment("Prod 1", config, tags=["production"])
        tracker.create_experiment("Prod 2", config, tags=["production", "v2"])
        tracker.create_experiment("Dev", config, tags=["development"])

        prod_experiments = tracker.list_experiments(tags=["production"])
        assert len(prod_experiments) == 2

    def test_delete_experiment(self, tracker):
        """Test deleting an experiment."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )
        exp = tracker.create_experiment("To Delete", config)
        exp_id = exp.experiment_id

        assert tracker.delete_experiment(exp_id) is True
        assert tracker.get_experiment(exp_id) is None
        assert tracker.delete_experiment(exp_id) is False  # Already deleted

    def test_compare_experiments(self, tracker):
        """Test comparing experiments."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        exp1 = tracker.create_experiment("Exp 1", config)
        tracker.start_experiment(exp1.experiment_id)
        tracker.complete_experiment(
            exp1.experiment_id,
            ExperimentResult(
                metrics={"sharpe": 1.5, "return": 0.20},
                trades_count=50,
            ),
        )

        exp2 = tracker.create_experiment("Exp 2", config)
        tracker.start_experiment(exp2.experiment_id)
        tracker.complete_experiment(
            exp2.experiment_id,
            ExperimentResult(
                metrics={"sharpe": 2.0, "return": 0.30},
                trades_count=75,
            ),
        )

        comparison = tracker.compare_experiments(
            [exp1.experiment_id, exp2.experiment_id],
            metrics=["sharpe", "return"],
        )

        assert len(comparison["experiments"]) == 2
        assert "metrics_comparison" in comparison
        assert comparison["rankings"]["sharpe"][0] == exp2.experiment_id

    def test_generate_comparison_report(self, tracker):
        """Test generating comparison report."""
        config = ExperimentConfig(
            strategy_name="grid",
            strategy_version="1.0.0",
            strategy_params={"count": 10},
            backtest_config={},
            data_source="binance",
            data_range=("2024-01-01", "2024-06-01"),
        )

        exp1 = tracker.create_experiment("Experiment A", config)
        tracker.start_experiment(exp1.experiment_id)
        tracker.complete_experiment(
            exp1.experiment_id,
            ExperimentResult(
                metrics={"sharpe": 1.2, "max_drawdown": 0.15},
                trades_count=100,
            ),
        )

        exp2 = tracker.create_experiment("Experiment B", config)
        tracker.start_experiment(exp2.experiment_id)
        tracker.complete_experiment(
            exp2.experiment_id,
            ExperimentResult(
                metrics={"sharpe": 1.8, "max_drawdown": 0.10},
                trades_count=80,
            ),
        )

        report = tracker.generate_comparison_report([exp1.experiment_id, exp2.experiment_id])

        assert "EXPERIMENT COMPARISON REPORT" in report
        assert "Experiment A" in report
        assert "Experiment B" in report

    def test_parent_experiment(self, tracker):
        """Test creating child experiment with parent reference."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        parent = tracker.create_experiment("Parent", config)
        child = tracker.create_experiment(
            "Child",
            config,
            parent_id=parent.experiment_id,
        )

        assert child.parent_id == parent.experiment_id

    def test_persistence(self, temp_dir):
        """Test experiments persist across tracker instances."""
        config = ExperimentConfig(
            strategy_name="test",
            strategy_version="1.0.0",
            strategy_params={"x": 1},
            backtest_config={},
            data_source="test",
            data_range=("2024-01-01", "2024-01-31"),
        )

        # Create and save with first tracker
        tracker1 = ExperimentTracker(base_path=temp_dir)
        exp = tracker1.create_experiment("Persistent", config, tags=["persist"])
        exp_id = exp.experiment_id

        # Load with second tracker
        tracker2 = ExperimentTracker(base_path=temp_dir)
        loaded = tracker2.get_experiment(exp_id)

        assert loaded is not None
        assert loaded.name == "Persistent"
        assert "persist" in loaded.tags

