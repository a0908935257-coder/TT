"""
Experiment Tracking Module.

Provides experiment management, result persistence, and comparison tools
for systematic strategy development and evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json
import hashlib
import uuid

from .versioning import DecimalEncoder, StrategyMetadata


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """
    Configuration for an experiment run.

    Attributes:
        strategy_name: Name of the strategy being tested
        strategy_version: Version of the strategy
        strategy_params: Strategy parameters
        backtest_config: Backtest configuration parameters
        data_source: Description of data used
        data_range: Date range of data (start, end)
        notes: Additional notes
    """

    strategy_name: str
    strategy_version: str = "1.0.0"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    backtest_config: dict[str, Any] = field(default_factory=dict)
    data_source: str = ""
    data_range: tuple[str, str] = ("", "")
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "strategy_params": self.strategy_params,
            "backtest_config": self.backtest_config,
            "data_source": self.data_source,
            "data_range": list(self.data_range),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            strategy_name=data["strategy_name"],
            strategy_version=data.get("strategy_version", "1.0.0"),
            strategy_params=data.get("strategy_params", {}),
            backtest_config=data.get("backtest_config", {}),
            data_source=data.get("data_source", ""),
            data_range=tuple(data.get("data_range", ["", ""])),
            notes=data.get("notes", ""),
        )


@dataclass
class ExperimentResult:
    """
    Results from an experiment run.

    Attributes:
        metrics: Key performance metrics
        trades_count: Number of trades executed
        equity_curve: List of equity values (optional, can be large)
        walk_forward_results: Walk-forward validation results
        optimization_results: Parameter optimization results
        error_message: Error message if failed
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    trades_count: int = 0
    equity_curve: Optional[list[float]] = None
    walk_forward_results: Optional[dict[str, Any]] = None
    optimization_results: Optional[dict[str, Any]] = None
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "trades_count": self.trades_count,
            "equity_curve": self.equity_curve,
            "walk_forward_results": self.walk_forward_results,
            "optimization_results": self.optimization_results,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(
            metrics=data.get("metrics", {}),
            trades_count=data.get("trades_count", 0),
            equity_curve=data.get("equity_curve"),
            walk_forward_results=data.get("walk_forward_results"),
            optimization_results=data.get("optimization_results"),
            error_message=data.get("error_message", ""),
        )


@dataclass
class Experiment:
    """
    Complete experiment record.

    Attributes:
        experiment_id: Unique experiment identifier
        name: Human-readable experiment name
        config: Experiment configuration
        result: Experiment results (None if not completed)
        status: Current status
        created_at: Creation timestamp
        started_at: Execution start timestamp
        completed_at: Completion timestamp
        tags: List of tags for categorization
        parent_id: Parent experiment ID (for derived experiments)
    """

    experiment_id: str
    name: str
    config: ExperimentConfig
    result: Optional[ExperimentResult] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "config": self.config.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            config=ExperimentConfig.from_dict(data["config"]),
            result=ExperimentResult.from_dict(data["result"]) if data.get("result") else None,
            status=ExperimentStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            tags=data.get("tags", []),
            parent_id=data.get("parent_id"),
        )


class ExperimentTracker:
    """
    Manages experiment lifecycle and persistence.

    Provides experiment creation, execution tracking, result storage,
    and comparison capabilities.

    Example:
        tracker = ExperimentTracker(base_path=Path("./experiments"))

        # Create an experiment
        exp = tracker.create_experiment(
            name="Grid Strategy Test",
            config=ExperimentConfig(
                strategy_name="grid",
                strategy_params={"grid_count": 10},
            )
        )

        # Start and complete
        tracker.start_experiment(exp.experiment_id)
        tracker.complete_experiment(exp.experiment_id, result)

        # Compare experiments
        comparison = tracker.compare_experiments([exp1.experiment_id, exp2.experiment_id])
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """
        Initialize experiment tracker.

        Args:
            base_path: Base directory for storing experiments
        """
        self._base_path = base_path or Path("./experiments")
        self._experiments_dir = self._base_path / "runs"
        self._index_file = self._base_path / "index.json"

        # Create directories
        self._experiments_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index: dict[str, dict[str, Any]] = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the experiment index."""
        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save the experiment index."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2, cls=DecimalEncoder)

    def _get_experiment_file(self, experiment_id: str) -> Path:
        """Get file path for an experiment."""
        return self._experiments_dir / f"{experiment_id}.json"

    def _generate_id(self) -> str:
        """Generate a unique experiment ID."""
        return str(uuid.uuid4())[:8]

    def create_experiment(
        self,
        name: str,
        config: ExperimentConfig,
        tags: Optional[list[str]] = None,
        parent_id: Optional[str] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Human-readable experiment name
            config: Experiment configuration
            tags: Optional tags for categorization
            parent_id: Optional parent experiment ID

        Returns:
            Created Experiment object
        """
        experiment = Experiment(
            experiment_id=self._generate_id(),
            name=name,
            config=config,
            tags=tags or [],
            parent_id=parent_id,
        )

        # Save experiment
        self._save_experiment(experiment)

        # Update index
        self._index[experiment.experiment_id] = {
            "name": experiment.name,
            "strategy_name": config.strategy_name,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "tags": experiment.tags,
        }
        self._save_index()

        return experiment

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to disk."""
        file_path = self._get_experiment_file(experiment.experiment_id)
        with open(file_path, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2, cls=DecimalEncoder)

    def load_experiment(self, experiment_id: str) -> Experiment:
        """
        Load an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object

        Raises:
            FileNotFoundError: If experiment not found
        """
        file_path = self._get_experiment_file(experiment_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        with open(file_path, "r") as f:
            data = json.load(f)

        return Experiment.from_dict(data)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get an experiment by ID, returning None if not found.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment object or None if not found
        """
        try:
            return self.load_experiment(experiment_id)
        except FileNotFoundError:
            return None

    def start_experiment(self, experiment_id: str) -> Experiment:
        """
        Mark an experiment as started.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated Experiment object
        """
        experiment = self.load_experiment(experiment_id)
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()

        self._save_experiment(experiment)
        self._index[experiment_id]["status"] = experiment.status.value
        self._save_index()

        return experiment

    def complete_experiment(
        self,
        experiment_id: str,
        result: ExperimentResult,
    ) -> Experiment:
        """
        Mark an experiment as completed with results.

        Args:
            experiment_id: Experiment ID
            result: Experiment results

        Returns:
            Updated Experiment object
        """
        experiment = self.load_experiment(experiment_id)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now()
        experiment.result = result

        self._save_experiment(experiment)
        self._index[experiment_id]["status"] = experiment.status.value
        self._index[experiment_id]["completed_at"] = experiment.completed_at.isoformat()

        # Add key metrics to index for quick access
        if result.metrics:
            self._index[experiment_id]["metrics_summary"] = {
                k: v
                for k, v in result.metrics.items()
                if k in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate"]
            }

        self._save_index()

        return experiment

    def fail_experiment(
        self,
        experiment_id: str,
        error_message: str,
    ) -> Experiment:
        """
        Mark an experiment as failed.

        Args:
            experiment_id: Experiment ID
            error_message: Error description

        Returns:
            Updated Experiment object
        """
        experiment = self.load_experiment(experiment_id)
        experiment.status = ExperimentStatus.FAILED
        experiment.completed_at = datetime.now()
        experiment.result = ExperimentResult(error_message=error_message)

        self._save_experiment(experiment)
        self._index[experiment_id]["status"] = experiment.status.value
        self._save_index()

        return experiment

    def list_experiments(
        self,
        strategy_name: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            strategy_name: Filter by strategy name
            status: Filter by status
            tags: Filter by tags (any match)
            limit: Maximum number to return

        Returns:
            List of experiment summaries
        """
        results = []

        for exp_id, exp_info in self._index.items():
            # Apply filters
            if strategy_name and exp_info.get("strategy_name") != strategy_name:
                continue
            if status and exp_info.get("status") != status.value:
                continue
            if tags and not any(t in exp_info.get("tags", []) for t in tags):
                continue

            results.append({"experiment_id": exp_id, **exp_info})

        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results[:limit]

    def compare_experiments(
        self,
        experiment_ids: list[str],
        metrics: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Comparison report dictionary
        """
        experiments = [self.load_experiment(eid) for eid in experiment_ids]

        # Collect all metrics
        all_metrics: dict[str, dict[str, Any]] = {}
        for exp in experiments:
            if exp.result and exp.result.metrics:
                exp_metrics = exp.result.metrics
                if metrics:
                    exp_metrics = {k: v for k, v in exp_metrics.items() if k in metrics}

                for key, value in exp_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = {}
                    all_metrics[key][exp.experiment_id] = value

        # Find best for each metric (assuming higher is better for most)
        rankings: dict[str, list[str]] = {}
        for metric_name, values in all_metrics.items():
            # Sort by value descending
            sorted_ids = sorted(
                values.keys(),
                key=lambda x: float(values[x]) if isinstance(values[x], (int, float, Decimal)) else 0,
                reverse=True,
            )
            rankings[metric_name] = sorted_ids

        return {
            "experiment_count": len(experiments),
            "experiments": [
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "config": exp.config.to_dict(),
                    "metrics": exp.result.metrics if exp.result else {},
                    "status": exp.status.value,
                }
                for exp in experiments
            ],
            "metrics_comparison": all_metrics,
            "rankings": rankings,
        }

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_experiment_file(experiment_id)
        if file_path.exists():
            file_path.unlink()

        if experiment_id in self._index:
            del self._index[experiment_id]
            self._save_index()
            return True

        return False

    def get_experiment_summary(self, experiment_id: str) -> str:
        """
        Generate a text summary of an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Formatted summary string
        """
        exp = self.load_experiment(experiment_id)

        lines = [
            "=" * 60,
            f"EXPERIMENT: {exp.name}",
            "=" * 60,
            f"ID: {exp.experiment_id}",
            f"Status: {exp.status.value}",
            f"Created: {exp.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if exp.duration_seconds:
            lines.append(f"Duration: {exp.duration_seconds:.2f}s")

        lines.extend(
            [
                "",
                "CONFIGURATION:",
                f"  Strategy: {exp.config.strategy_name} v{exp.config.strategy_version}",
                f"  Data: {exp.config.data_source}",
            ]
        )

        if exp.config.strategy_params:
            lines.append("  Parameters:")
            for k, v in exp.config.strategy_params.items():
                lines.append(f"    {k}: {v}")

        if exp.result and exp.result.metrics:
            lines.extend(["", "RESULTS:"])
            for k, v in exp.result.metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

            if exp.result.trades_count:
                lines.append(f"  Trades: {exp.result.trades_count}")

        if exp.result and exp.result.error_message:
            lines.extend(["", f"ERROR: {exp.result.error_message}"])

        if exp.tags:
            lines.extend(["", f"Tags: {', '.join(exp.tags)}"])

        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_comparison_report(
        self,
        experiment_ids: list[str],
        output_format: str = "text",
    ) -> str:
        """
        Generate a comparison report for multiple experiments.

        Args:
            experiment_ids: List of experiment IDs
            output_format: Output format ('text' or 'markdown')

        Returns:
            Formatted report string
        """
        comparison = self.compare_experiments(experiment_ids)

        if output_format == "markdown":
            return self._generate_markdown_report(comparison)
        else:
            return self._generate_text_report(comparison)

    def _generate_text_report(self, comparison: dict[str, Any]) -> str:
        """Generate text comparison report."""
        lines = [
            "=" * 80,
            "EXPERIMENT COMPARISON REPORT",
            "=" * 80,
            f"Comparing {comparison['experiment_count']} experiments",
            "",
        ]

        # Experiment details
        for exp in comparison["experiments"]:
            lines.append(f"[{exp['id']}] {exp['name']}")
            lines.append(f"    Strategy: {exp['config']['strategy_name']}")
            lines.append(f"    Status: {exp['status']}")
            lines.append("")

        # Metrics comparison table
        lines.extend(["", "METRICS COMPARISON:", "-" * 60])

        metrics = comparison["metrics_comparison"]
        exp_ids = [e["id"] for e in comparison["experiments"]]

        # Header
        header = "Metric".ljust(25) + " | " + " | ".join(eid[:8].center(12) for eid in exp_ids)
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for metric_name, values in metrics.items():
            row = metric_name[:24].ljust(25) + " | "
            for eid in exp_ids:
                val = values.get(eid, "N/A")
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                else:
                    val_str = str(val)
                row += val_str[:12].center(12) + " | "
            lines.append(row)

        # Rankings
        lines.extend(["", "RANKINGS (by metric):", "-" * 40])
        for metric_name, ranking in comparison["rankings"].items():
            lines.append(f"  {metric_name}: {' > '.join(r[:8] for r in ranking)}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def _generate_markdown_report(self, comparison: dict[str, Any]) -> str:
        """Generate markdown comparison report."""
        lines = [
            "# Experiment Comparison Report",
            "",
            f"Comparing {comparison['experiment_count']} experiments",
            "",
            "## Experiments",
            "",
        ]

        for exp in comparison["experiments"]:
            lines.extend(
                [
                    f"### {exp['name']} (`{exp['id']}`)",
                    f"- **Strategy**: {exp['config']['strategy_name']}",
                    f"- **Status**: {exp['status']}",
                    "",
                ]
            )

        # Metrics table
        lines.extend(["## Metrics Comparison", ""])

        metrics = comparison["metrics_comparison"]
        exp_ids = [e["id"] for e in comparison["experiments"]]

        # Table header
        lines.append("| Metric | " + " | ".join(eid[:8] for eid in exp_ids) + " |")
        lines.append("|" + "---|" * (len(exp_ids) + 1))

        # Table rows
        for metric_name, values in metrics.items():
            row = f"| {metric_name} |"
            for eid in exp_ids:
                val = values.get(eid, "N/A")
                if isinstance(val, float):
                    row += f" {val:.4f} |"
                else:
                    row += f" {val} |"
            lines.append(row)

        return "\n".join(lines)
