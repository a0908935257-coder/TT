"""
Strategy Version Control Module.

Provides serialization, metadata management, and version tracking
for trading strategies.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Type
import hashlib
import json
import copy

# Try to import yaml
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class SerializationFormat(str, Enum):
    """Supported serialization formats."""

    JSON = "json"
    YAML = "yaml"


@dataclass
class StrategyMetadata:
    """
    Metadata for a strategy version.

    Attributes:
        name: Strategy name
        version: Version string (semver recommended)
        description: Strategy description
        author: Author name or identifier
        created_at: Creation timestamp
        updated_at: Last update timestamp
        tags: List of tags for categorization
        parent_version: Version this was derived from
        notes: Additional notes or changelog
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "parent_version": self.parent_version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(timezone.utc),
            tags=data.get("tags", []),
            parent_version=data.get("parent_version"),
            notes=data.get("notes", ""),
        )


@dataclass
class StrategySnapshot:
    """
    Complete snapshot of a strategy configuration.

    Attributes:
        metadata: Strategy metadata
        config: Strategy configuration parameters
        config_class: Fully qualified class name of the config
        checksum: Hash of the configuration for integrity
    """

    metadata: StrategyMetadata
    config: dict[str, Any]
    config_class: str = ""
    checksum: str = ""

    def __post_init__(self) -> None:
        """Calculate checksum if not provided."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate MD5 checksum of configuration."""
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify configuration integrity."""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "config": self.config,
            "config_class": self.config_class,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategySnapshot":
        """Create from dictionary."""
        return cls(
            metadata=StrategyMetadata.from_dict(data["metadata"]),
            config=data["config"],
            config_class=data.get("config_class", ""),
            checksum=data.get("checksum", ""),
        )


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def decimal_decoder(dct: dict) -> dict:
    """Decode Decimal strings back to Decimal objects."""
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                # Try to parse as Decimal if it looks like a number
                if value.replace(".", "").replace("-", "").isdigit():
                    dct[key] = Decimal(value)
            except Exception:
                pass
    return dct


class StrategySerializer:
    """
    Serializer for strategy configurations.

    Supports JSON and YAML formats with proper handling of
    Decimal, datetime, and Enum types.

    Example:
        serializer = StrategySerializer()

        # Serialize a config to JSON
        json_str = serializer.serialize(config, format=SerializationFormat.JSON)

        # Deserialize back
        config = serializer.deserialize(json_str, MyConfigClass, format=SerializationFormat.JSON)
    """

    def serialize(
        self,
        config: Any,
        format: SerializationFormat = SerializationFormat.JSON,
        metadata: Optional[StrategyMetadata] = None,
        indent: int = 2,
    ) -> str:
        """
        Serialize a configuration object.

        Args:
            config: Configuration object (dataclass or dict)
            format: Output format (JSON or YAML)
            metadata: Optional metadata to include
            indent: Indentation level

        Returns:
            Serialized string
        """
        # Convert to dict if dataclass
        if hasattr(config, "__dataclass_fields__"):
            config_dict = self._dataclass_to_dict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(f"Cannot serialize type: {type(config)}")

        # Build output structure
        if metadata:
            output = {
                "metadata": metadata.to_dict(),
                "config": config_dict,
            }
        else:
            output = config_dict

        # Serialize
        if format == SerializationFormat.JSON:
            return json.dumps(output, cls=DecimalEncoder, indent=indent)
        elif format == SerializationFormat.YAML:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML format. Install with: pip install pyyaml")
            # Convert Decimals to strings for YAML
            output = self._convert_decimals_to_str(output)
            return yaml.dump(output, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def deserialize(
        self,
        data: str,
        config_class: Optional[Type] = None,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> tuple[Any, Optional[StrategyMetadata]]:
        """
        Deserialize a configuration string.

        Args:
            data: Serialized string
            config_class: Optional class to instantiate (otherwise returns dict)
            format: Input format

        Returns:
            Tuple of (config, metadata) where metadata may be None
        """
        # Parse
        if format == SerializationFormat.JSON:
            parsed = json.loads(data, object_hook=decimal_decoder)
        elif format == SerializationFormat.YAML:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML format")
            parsed = yaml.safe_load(data)
            parsed = self._convert_str_to_decimals(parsed)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Extract metadata if present
        metadata = None
        config_dict = parsed

        if isinstance(parsed, dict) and "metadata" in parsed and "config" in parsed:
            metadata = StrategyMetadata.from_dict(parsed["metadata"])
            config_dict = parsed["config"]

        # Instantiate class if provided
        if config_class and hasattr(config_class, "__dataclass_fields__"):
            config = self._dict_to_dataclass(config_dict, config_class)
        else:
            config = config_dict

        return config, metadata

    def _dataclass_to_dict(self, obj: Any) -> dict:
        """Convert a dataclass to a dictionary recursively."""
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                result[field_name] = self._convert_value(value)
            return result
        return obj

    def _convert_value(self, value: Any) -> Any:
        """Convert a value for serialization."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif hasattr(value, "__dataclass_fields__"):
            return self._dataclass_to_dict(value)
        elif isinstance(value, list):
            return [self._convert_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        return value

    def _dict_to_dataclass(self, data: dict, cls: Type) -> Any:
        """Convert a dictionary to a dataclass instance."""
        if not hasattr(cls, "__dataclass_fields__"):
            return data

        field_values = {}
        for field_name, field_info in cls.__dataclass_fields__.items():
            if field_name not in data:
                continue

            value = data[field_name]
            field_type = field_info.type

            # Handle Decimal
            if field_type == Decimal or (
                hasattr(field_type, "__origin__") and field_type.__origin__ is Decimal
            ):
                value = Decimal(str(value)) if value is not None else None
            # Handle datetime
            elif field_type == datetime:
                if isinstance(value, str):
                    value = datetime.fromisoformat(value)
            # Handle nested dataclass
            elif hasattr(field_type, "__dataclass_fields__"):
                value = self._dict_to_dataclass(value, field_type)

            field_values[field_name] = value

        return cls(**field_values)

    def _convert_decimals_to_str(self, obj: Any) -> Any:
        """Convert Decimal values to strings recursively."""
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_decimals_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals_to_str(v) for v in obj]
        return obj

    def _convert_str_to_decimals(self, obj: Any) -> Any:
        """Convert string numbers to Decimal recursively."""
        if isinstance(obj, str):
            try:
                if obj.replace(".", "").replace("-", "").isdigit():
                    return Decimal(obj)
            except Exception:
                pass
            return obj
        elif isinstance(obj, dict):
            return {k: self._convert_str_to_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_str_to_decimals(v) for v in obj]
        return obj


class VersionManager:
    """
    Manages strategy versions and snapshots.

    Provides version history tracking, comparison, and rollback capabilities.

    Example:
        manager = VersionManager(base_path=Path("./strategies"))

        # Save a strategy version
        snapshot = manager.save_version(config, metadata)

        # Load a specific version
        config = manager.load_version("my_strategy", "1.0.0")

        # List all versions
        versions = manager.list_versions("my_strategy")
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        format: SerializationFormat = SerializationFormat.JSON,
    ) -> None:
        """
        Initialize version manager.

        Args:
            base_path: Base directory for storing versions
            format: Serialization format to use
        """
        self._base_path = base_path or Path("./strategies")
        self._format = format
        self._serializer = StrategySerializer()

        # Create base directory if needed
        if self._base_path:
            self._base_path.mkdir(parents=True, exist_ok=True)

    def _get_strategy_dir(self, strategy_name: str) -> Path:
        """Get directory for a strategy."""
        return self._base_path / strategy_name

    def _get_version_file(self, strategy_name: str, version: str) -> Path:
        """Get file path for a specific version."""
        ext = "json" if self._format == SerializationFormat.JSON else "yaml"
        return self._get_strategy_dir(strategy_name) / f"v{version}.{ext}"

    def save_version(
        self,
        config: Any,
        metadata: StrategyMetadata,
    ) -> StrategySnapshot:
        """
        Save a strategy version.

        Args:
            config: Strategy configuration
            metadata: Strategy metadata

        Returns:
            StrategySnapshot of the saved version
        """
        # Update timestamp
        metadata.updated_at = datetime.now(timezone.utc)

        # Create snapshot
        if hasattr(config, "__dataclass_fields__"):
            config_dict = self._serializer._dataclass_to_dict(config)
            config_class = f"{config.__class__.__module__}.{config.__class__.__name__}"
        else:
            config_dict = config
            config_class = ""

        snapshot = StrategySnapshot(
            metadata=metadata,
            config=config_dict,
            config_class=config_class,
        )

        # Create directory if needed
        strategy_dir = self._get_strategy_dir(metadata.name)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = self._get_version_file(metadata.name, metadata.version)
        serialized = self._serializer.serialize(
            snapshot.to_dict(),
            format=self._format,
        )

        with open(file_path, "w") as f:
            f.write(serialized)

        # Update latest symlink/reference
        self._update_latest(metadata.name, metadata.version)

        return snapshot

    def load_version(
        self,
        strategy_name: str,
        version: Optional[str] = None,
        config_class: Optional[Type] = None,
    ) -> tuple[Any, StrategyMetadata]:
        """
        Load a strategy version.

        Args:
            strategy_name: Name of the strategy
            version: Version to load (None = latest)
            config_class: Class to deserialize into

        Returns:
            Tuple of (config, metadata)
        """
        if version is None:
            version = self._get_latest_version(strategy_name)

        file_path = self._get_version_file(strategy_name, version)

        if not file_path.exists():
            raise FileNotFoundError(f"Version {version} not found for strategy {strategy_name}")

        with open(file_path, "r") as f:
            data = f.read()

        # Parse raw file content (snapshot format, not serializer format)
        if self._format == SerializationFormat.JSON:
            parsed = json.loads(data)
        else:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML format")
            import yaml
            parsed = yaml.safe_load(data)

        # Parse as snapshot
        snapshot = StrategySnapshot.from_dict(parsed)

        # Verify integrity
        if not snapshot.verify_integrity():
            raise ValueError(f"Configuration integrity check failed for {strategy_name} v{version}")

        # Convert config to class if provided
        if config_class:
            config = self._serializer._dict_to_dataclass(snapshot.config, config_class)
        else:
            config = snapshot.config

        return config, snapshot.metadata

    def list_versions(self, strategy_name: str) -> list[str]:
        """
        List all versions of a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            List of version strings, sorted newest first
        """
        strategy_dir = self._get_strategy_dir(strategy_name)
        if not strategy_dir.exists():
            return []

        versions = []
        ext = "json" if self._format == SerializationFormat.JSON else "yaml"

        for file in strategy_dir.glob(f"v*.{ext}"):
            # Extract version from filename
            version = file.stem[1:]  # Remove 'v' prefix
            versions.append(version)

        # Sort by semver (simplified)
        versions.sort(key=self._parse_version, reverse=True)
        return versions

    def _parse_version(self, version: str) -> tuple:
        """Parse version string for sorting."""
        try:
            parts = version.split(".")
            return tuple(int(p) for p in parts)
        except ValueError:
            return (0, 0, 0)

    def _get_latest_version(self, strategy_name: str) -> str:
        """Get the latest version of a strategy."""
        versions = self.list_versions(strategy_name)
        if not versions:
            raise FileNotFoundError(f"No versions found for strategy {strategy_name}")
        return versions[0]

    def _update_latest(self, strategy_name: str, version: str) -> None:
        """Update the latest version reference."""
        strategy_dir = self._get_strategy_dir(strategy_name)
        latest_file = strategy_dir / "latest.txt"
        with open(latest_file, "w") as f:
            f.write(version)

    def compare_versions(
        self,
        strategy_name: str,
        version1: str,
        version2: str,
    ) -> dict[str, Any]:
        """
        Compare two versions of a strategy.

        Args:
            strategy_name: Name of the strategy
            version1: First version
            version2: Second version

        Returns:
            Dictionary with comparison results
        """
        config1, meta1 = self.load_version(strategy_name, version1)
        config2, meta2 = self.load_version(strategy_name, version2)

        # Find differences
        differences = self._find_differences(config1, config2)

        return {
            "version1": version1,
            "version2": version2,
            "version1_date": meta1.updated_at.isoformat(),
            "version2_date": meta2.updated_at.isoformat(),
            "differences": differences,
            "n_differences": len(differences),
        }

    def _find_differences(
        self,
        obj1: Any,
        obj2: Any,
        path: str = "",
    ) -> list[dict[str, Any]]:
        """Find differences between two objects."""
        differences = []

        if type(obj1) != type(obj2):
            differences.append(
                {
                    "path": path or "root",
                    "type": "type_change",
                    "old": str(type(obj1).__name__),
                    "new": str(type(obj2).__name__),
                }
            )
            return differences

        if isinstance(obj1, dict):
            all_keys = set(obj1.keys()) | set(obj2.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                if key not in obj1:
                    differences.append(
                        {"path": key_path, "type": "added", "new": obj2[key]}
                    )
                elif key not in obj2:
                    differences.append(
                        {"path": key_path, "type": "removed", "old": obj1[key]}
                    )
                else:
                    differences.extend(
                        self._find_differences(obj1[key], obj2[key], key_path)
                    )
        elif obj1 != obj2:
            differences.append(
                {"path": path or "root", "type": "changed", "old": obj1, "new": obj2}
            )

        return differences

    def delete_version(self, strategy_name: str, version: str) -> bool:
        """
        Delete a specific version.

        Args:
            strategy_name: Name of the strategy
            version: Version to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_version_file(strategy_name, version)
        if file_path.exists():
            file_path.unlink()
            return True
        return False


def is_yaml_available() -> bool:
    """Check if PyYAML is available."""
    return YAML_AVAILABLE
