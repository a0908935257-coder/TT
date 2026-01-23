"""
Unit tests for strategy versioning module.
"""

import pytest
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.backtest.versioning import (
    StrategyMetadata,
    StrategySnapshot,
    StrategySerializer,
    VersionManager,
    SerializationFormat,
    is_yaml_available,
)


@dataclass
class SampleStrategyConfig:
    """Sample strategy config for testing."""

    name: str = "test_strategy"
    grid_count: int = 10
    range_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))
    enabled: bool = True
    tags: list = field(default_factory=list)


class TestStrategyMetadata:
    """Tests for StrategyMetadata class."""

    def test_default_metadata(self):
        """Test default metadata creation."""
        meta = StrategyMetadata(name="my_strategy")
        assert meta.name == "my_strategy"
        assert meta.version == "1.0.0"
        assert meta.description == ""
        assert isinstance(meta.created_at, datetime)

    def test_custom_metadata(self):
        """Test custom metadata creation."""
        meta = StrategyMetadata(
            name="my_strategy",
            version="2.1.0",
            description="A test strategy",
            author="test_user",
            tags=["experimental", "grid"],
        )
        assert meta.version == "2.1.0"
        assert meta.description == "A test strategy"
        assert "grid" in meta.tags

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = StrategyMetadata(
            name="test",
            version="1.0.0",
            tags=["a", "b"],
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert d["tags"] == ["a", "b"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "version": "1.2.3",
            "description": "desc",
            "author": "me",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T12:00:00",
            "tags": ["x"],
            "notes": "some notes",
        }
        meta = StrategyMetadata.from_dict(data)
        assert meta.name == "test"
        assert meta.version == "1.2.3"
        assert meta.author == "me"
        assert meta.notes == "some notes"


class TestStrategySnapshot:
    """Tests for StrategySnapshot class."""

    def test_checksum_calculation(self):
        """Test checksum is calculated on creation."""
        snapshot = StrategySnapshot(
            metadata=StrategyMetadata(name="test"),
            config={"a": 1, "b": 2},
        )
        assert len(snapshot.checksum) == 32  # MD5 hex

    def test_verify_integrity_valid(self):
        """Test integrity verification passes for valid config."""
        snapshot = StrategySnapshot(
            metadata=StrategyMetadata(name="test"),
            config={"x": 1},
        )
        assert snapshot.verify_integrity() is True

    def test_verify_integrity_invalid(self):
        """Test integrity verification fails after tampering."""
        snapshot = StrategySnapshot(
            metadata=StrategyMetadata(name="test"),
            config={"x": 1},
        )
        # Tamper with config
        snapshot.config["x"] = 999
        assert snapshot.verify_integrity() is False

    def test_to_dict_and_from_dict(self):
        """Test roundtrip serialization."""
        original = StrategySnapshot(
            metadata=StrategyMetadata(name="test", version="1.0.0"),
            config={"grid_count": 10, "rate": "0.05"},
            config_class="SampleConfig",
        )
        d = original.to_dict()
        restored = StrategySnapshot.from_dict(d)

        assert restored.metadata.name == original.metadata.name
        assert restored.config == original.config
        assert restored.config_class == original.config_class
        assert restored.checksum == original.checksum


class TestStrategySerializer:
    """Tests for StrategySerializer class."""

    @pytest.fixture
    def serializer(self):
        """Create a serializer instance."""
        return StrategySerializer()

    def test_serialize_dict_json(self, serializer):
        """Test serializing a dict to JSON."""
        config = {"name": "test", "count": 10}
        json_str = serializer.serialize(config, format=SerializationFormat.JSON)
        assert '"name": "test"' in json_str
        assert '"count": 10' in json_str

    def test_serialize_dataclass_json(self, serializer):
        """Test serializing a dataclass to JSON."""
        config = SampleStrategyConfig(
            name="grid",
            grid_count=15,
            range_pct=Decimal("0.1"),
        )
        json_str = serializer.serialize(config, format=SerializationFormat.JSON)
        assert '"name": "grid"' in json_str
        assert '"grid_count": 15' in json_str
        # Decimal should be string
        assert '"range_pct": "0.1"' in json_str

    def test_serialize_with_metadata(self, serializer):
        """Test serializing with metadata."""
        config = {"x": 1}
        meta = StrategyMetadata(name="my_strat", version="1.2.0")
        json_str = serializer.serialize(
            config,
            format=SerializationFormat.JSON,
            metadata=meta,
        )
        assert '"metadata":' in json_str
        assert '"config":' in json_str
        assert '"my_strat"' in json_str

    def test_deserialize_dict_json(self, serializer):
        """Test deserializing JSON to dict."""
        json_str = '{"name": "test", "count": 10}'
        config, meta = serializer.deserialize(
            json_str,
            format=SerializationFormat.JSON,
        )
        assert config["name"] == "test"
        assert config["count"] == 10
        assert meta is None

    def test_deserialize_with_metadata(self, serializer):
        """Test deserializing JSON with metadata."""
        json_str = '''
        {
            "metadata": {
                "name": "test",
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00"
            },
            "config": {"x": 100}
        }
        '''
        config, meta = serializer.deserialize(
            json_str,
            format=SerializationFormat.JSON,
        )
        assert config["x"] == 100
        assert meta is not None
        assert meta.name == "test"

    @pytest.mark.skipif(not is_yaml_available(), reason="PyYAML not installed")
    def test_serialize_yaml(self, serializer):
        """Test serializing to YAML."""
        config = {"name": "test", "count": 10}
        yaml_str = serializer.serialize(config, format=SerializationFormat.YAML)
        assert "name:" in yaml_str
        assert "count:" in yaml_str

    @pytest.mark.skipif(not is_yaml_available(), reason="PyYAML not installed")
    def test_deserialize_yaml(self, serializer):
        """Test deserializing YAML."""
        yaml_str = "name: test\ncount: 10"
        config, meta = serializer.deserialize(
            yaml_str,
            format=SerializationFormat.YAML,
        )
        assert config["name"] == "test"
        assert config["count"] == 10


class TestVersionManager:
    """Tests for VersionManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a version manager with temp directory."""
        return VersionManager(base_path=temp_dir)

    def test_save_and_load_version(self, manager):
        """Test saving and loading a version."""
        config = {"grid_count": 10, "rate": 0.05}
        meta = StrategyMetadata(name="grid_strategy", version="1.0.0")

        # Save
        snapshot = manager.save_version(config, meta)
        assert snapshot.checksum != ""

        # Load
        loaded_config, loaded_meta = manager.load_version("grid_strategy", "1.0.0")
        assert loaded_config["grid_count"] == 10
        assert loaded_meta.name == "grid_strategy"

    def test_load_latest_version(self, manager):
        """Test loading latest version."""
        meta1 = StrategyMetadata(name="test", version="1.0.0")
        meta2 = StrategyMetadata(name="test", version="2.0.0")

        manager.save_version({"v": 1}, meta1)
        manager.save_version({"v": 2}, meta2)

        config, meta = manager.load_version("test")  # No version = latest
        assert config["v"] == 2
        assert meta.version == "2.0.0"

    def test_list_versions(self, manager):
        """Test listing versions."""
        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            meta = StrategyMetadata(name="test", version=v)
            manager.save_version({"v": v}, meta)

        versions = manager.list_versions("test")
        assert len(versions) == 3
        # Should be sorted newest first
        assert versions[0] == "2.0.0"
        assert versions[1] == "1.1.0"
        assert versions[2] == "1.0.0"

    def test_compare_versions(self, manager):
        """Test version comparison."""
        meta1 = StrategyMetadata(name="test", version="1.0.0")
        meta2 = StrategyMetadata(name="test", version="2.0.0")

        manager.save_version({"a": 1, "b": 2}, meta1)
        manager.save_version({"a": 1, "b": 3, "c": 4}, meta2)

        comparison = manager.compare_versions("test", "1.0.0", "2.0.0")

        assert comparison["version1"] == "1.0.0"
        assert comparison["version2"] == "2.0.0"
        assert len(comparison["differences"]) > 0

    def test_delete_version(self, manager):
        """Test deleting a version."""
        meta = StrategyMetadata(name="test", version="1.0.0")
        manager.save_version({"x": 1}, meta)

        assert manager.delete_version("test", "1.0.0") is True
        assert manager.delete_version("test", "1.0.0") is False  # Already deleted

    def test_integrity_check_on_load(self, manager):
        """Test that integrity is verified on load."""
        meta = StrategyMetadata(name="test", version="1.0.0")
        snapshot = manager.save_version({"x": 1}, meta)

        # Manually corrupt the file
        file_path = manager._get_version_file("test", "1.0.0")
        with open(file_path, "r") as f:
            data = f.read()
        # Change the config value but keep the old checksum
        corrupted = data.replace('"x": 1', '"x": 999')
        with open(file_path, "w") as f:
            f.write(corrupted)

        with pytest.raises(ValueError, match="integrity check failed"):
            manager.load_version("test", "1.0.0")
