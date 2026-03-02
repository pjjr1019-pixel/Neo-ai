"""Tests for data and model lineage tracking module."""

import pytest

from python_ai.data_lineage import (
    LineageRecord,
    LineageStore,
    hash_bytes,
    hash_file,
)


# ── LineageRecord ──────────────────────────────────────────


class TestLineageRecord:
    """Tests for the LineageRecord dataclass."""

    def test_defaults(self):
        """Record has sensible defaults."""
        rec = LineageRecord(
            record_id="abc",
            record_type="data",
            name="test",
        )
        assert rec.version == "1.0"
        assert rec.parent_ids == []
        assert rec.metadata == {}

    def test_to_dict(self):
        """to_dict produces a JSON-friendly dict."""
        rec = LineageRecord(
            record_id="id1",
            record_type="data",
            name="foo",
            data_hash="abc123",
        )
        d = rec.to_dict()
        assert d["record_id"] == "id1"
        assert d["name"] == "foo"

    def test_from_dict(self):
        """from_dict reconstructs a record."""
        d = {
            "record_id": "id2",
            "record_type": "transform",
            "name": "norm",
            "version": "2.0",
            "created_at": 1000.0,
            "data_hash": "h",
            "parent_ids": ["id1"],
            "metadata": {"k": "v"},
        }
        rec = LineageRecord.from_dict(d)
        assert rec.record_type == "transform"
        assert rec.parent_ids == ["id1"]
        assert rec.metadata["k"] == "v"

    def test_roundtrip(self):
        """to_dict -> from_dict is lossless."""
        rec = LineageRecord(
            record_id="rt",
            record_type="model",
            name="m",
            version="3.0",
            created_at=42.0,
            data_hash="h",
            parent_ids=["p1", "p2"],
            metadata={"x": 1},
        )
        rec2 = LineageRecord.from_dict(rec.to_dict())
        assert rec == rec2


# ── LineageStore ───────────────────────────────────────────


class TestLineageStore:
    """Tests for the file-backed lineage store."""

    @pytest.fixture()
    def store(self, tmp_path):
        """Create a fresh store in a temp directory."""
        path = str(tmp_path / "lineage.json")
        return LineageStore(store_path=path)

    def test_record_data(self, store):
        """record_data creates a data record."""
        rec = store.record_data(
            name="ohlcv",
            data_hash="aaa",
        )
        assert rec.record_type == "data"
        assert rec.name == "ohlcv"

    def test_record_transform(self, store):
        """record_transform creates a transform record."""
        d = store.record_data("src", "h1")
        t = store.record_transform(
            name="normalize",
            input_ids=[d.record_id],
            output_hash="h2",
        )
        assert t.record_type == "transform"
        assert d.record_id in t.parent_ids

    def test_record_model(self, store):
        """record_model creates a model record."""
        d = store.record_data("train", "h1")
        m = store.record_model(
            name="neo_v1",
            data_ids=[d.record_id],
            model_hash="mh",
        )
        assert m.record_type == "model"
        assert d.record_id in m.parent_ids

    def test_get(self, store):
        """get retrieves a record by ID."""
        rec = store.record_data("ds", "hh")
        found = store.get(rec.record_id)
        assert found is not None
        assert found.name == "ds"

    def test_get_missing(self, store):
        """get returns None for unknown ID."""
        assert store.get("nonexistent") is None

    def test_list_records(self, store):
        """list_records returns all records."""
        store.record_data("a", "h1")
        store.record_data("b", "h2")
        store.record_transform("t", ["x"], "h3")
        all_recs = store.list_records()
        assert len(all_recs) == 3

    def test_list_records_filtered(self, store):
        """list_records filters by type."""
        store.record_data("a", "h1")
        store.record_transform("t", [], "h2")
        data_only = store.list_records("data")
        assert len(data_only) == 1
        assert data_only[0].record_type == "data"

    def test_summary(self, store):
        """summary returns counts by type."""
        store.record_data("a", "h1")
        store.record_model("m", [], "h2")
        s = store.summary()
        assert s["total_records"] == 2
        assert s["by_type"]["data"] == 1
        assert s["by_type"]["model"] == 1

    def test_get_ancestors(self, store):
        """get_ancestors returns full upstream chain."""
        d = store.record_data("raw", "h0")
        t = store.record_transform(
            "clean", [d.record_id], "h1"
        )
        m = store.record_model(
            "model", [t.record_id], "h2"
        )
        ancestors = store.get_ancestors(m.record_id)
        ancestor_ids = {a.record_id for a in ancestors}
        assert d.record_id in ancestor_ids
        assert t.record_id in ancestor_ids

    def test_persistence(self, tmp_path):
        """Store persists and reloads records."""
        path = str(tmp_path / "lineage.json")
        s1 = LineageStore(store_path=path)
        rec = s1.record_data("persist", "ph")

        s2 = LineageStore(store_path=path)
        found = s2.get(rec.record_id)
        assert found is not None
        assert found.name == "persist"


# ── hash helpers ───────────────────────────────────────────


class TestHashHelpers:
    """Tests for hash_bytes and hash_file."""

    def test_hash_bytes_deterministic(self):
        """Same input produces same hash."""
        h1 = hash_bytes(b"test data")
        h2 = hash_bytes(b"test data")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_hash_bytes_different(self):
        """Different input produces different hash."""
        h1 = hash_bytes(b"aaa")
        h2 = hash_bytes(b"bbb")
        assert h1 != h2

    def test_hash_file(self, tmp_path):
        """hash_file hashes file contents."""
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        h = hash_file(str(p))
        assert len(h) == 64
        assert h == hash_bytes(b"hello world")
