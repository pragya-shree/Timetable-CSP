"""Tests for persistent storage module (core/storage.py).

Validates saving, loading, listing, duplicating, renaming, deleting,
corruption recovery, and model reconstruction in isolated temporary directories.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.models import load_dataset_from_file
from core.backtracking import solve_backtracking
from core.storage import (
    TimetableRecord,
    TimetableStore,
    TimetableNotFoundError,
    CorruptStorageFileError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_timetable():
    ds = load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")
    res = solve_backtracking(ds, use_mrv=True)
    assert res.success
    return res.timetable


class TestTimetableStorage:
    def test_save_and_load_record(self, tmp_path):
        store = TimetableStore(tmp_path)
        ds = load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")
        res = solve_backtracking(ds, use_mrv=True)

        rec = TimetableRecord.create_from_timetable(
            timetable=res.timetable,
            name="Test Schedule 1",
            algorithm="MRV Backtracking",
            runtime_seconds=0.012,
            search_metrics={"nodes_explored": 40},
        )
        saved = store.save(rec)
        assert saved.id == rec.id
        assert (tmp_path / f"{rec.id}.json").exists()

        # Load back
        loaded = store.get(rec.id)
        assert loaded.id == rec.id
        assert loaded.name == "Test Schedule 1"
        assert loaded.algorithm == "MRV Backtracking"
        assert loaded.validation["is_valid"] is True
        assert loaded.cost["hard_violations"] == 0

        # Model reconstruction
        reconstructed_tt = loaded.get_timetable()
        assert reconstructed_tt.is_complete()
        assert len(reconstructed_tt.get_placed_sessions()) == len(res.timetable.get_placed_sessions())
        assert reconstructed_tt.dataset.name == ds.name

    def test_save_multiple_without_overwriting(self, tmp_path, sample_timetable):
        store = TimetableStore(tmp_path)

        rec1 = TimetableRecord.create_from_timetable(sample_timetable, "Schedule Alpha", "MRV")
        rec2 = TimetableRecord.create_from_timetable(sample_timetable, "Schedule Beta", "Plain")

        store.save(rec1)
        store.save(rec2)

        all_records = store.list_all()
        assert len(all_records) == 2
        ids = {r.id for r in all_records}
        assert rec1.id in ids
        assert rec2.id in ids

    def test_rename_timetable(self, tmp_path, sample_timetable):
        store = TimetableStore(tmp_path)
        rec = TimetableRecord.create_from_timetable(sample_timetable, "Old Name", "MRV")
        store.save(rec)

        updated = store.rename(rec.id, "New Brand Name")
        assert updated.name == "New Brand Name"
        assert updated.version == 2

        # Check re-read from disk
        re_read = store.get(rec.id)
        assert re_read.name == "New Brand Name"
        assert re_read.version == 2

    def test_duplicate_timetable(self, tmp_path, sample_timetable):
        store = TimetableStore(tmp_path)
        rec = TimetableRecord.create_from_timetable(sample_timetable, "Original", "MRV")
        store.save(rec)

        dup = store.duplicate(rec.id, "Original Duplicated")
        assert dup.id != rec.id
        assert dup.name == "Original Duplicated"

        all_records = store.list_all()
        assert len(all_records) == 2
        # Verify both can be loaded independently
        t1 = store.get(rec.id)
        t2 = store.get(dup.id)
        assert t1.name == "Original"
        assert t2.name == "Original Duplicated"

    def test_delete_timetable(self, tmp_path, sample_timetable):
        store = TimetableStore(tmp_path)
        rec = TimetableRecord.create_from_timetable(sample_timetable, "To Delete", "MRV")
        store.save(rec)
        assert (tmp_path / f"{rec.id}.json").exists()

        deleted = store.delete(rec.id)
        assert deleted is True
        assert not (tmp_path / f"{rec.id}.json").exists()

        with pytest.raises(TimetableNotFoundError):
            store.get(rec.id)

    def test_missing_timetable_raises(self, tmp_path):
        store = TimetableStore(tmp_path)
        with pytest.raises(TimetableNotFoundError):
            store.get("non_existent_id")

    def test_corrupt_file_handling(self, tmp_path, sample_timetable):
        store = TimetableStore(tmp_path)
        # Create a valid record
        rec = TimetableRecord.create_from_timetable(sample_timetable, "Valid One", "MRV")
        store.save(rec)

        # Create a corrupt JSON file
        corrupt_file = tmp_path / "corrupt_record.json"
        corrupt_file.write_text("{ this is not valid json! }", encoding="utf-8")

        # Create another file with missing required fields
        invalid_schema_file = tmp_path / "bad_schema.json"
        invalid_schema_file.write_text(json.dumps({"some_key": "some_value"}), encoding="utf-8")

        # list_all should skip both corrupt files gracefully and return the valid one
        records = store.list_all()
        assert len(records) == 1
        assert records[0].id == rec.id

        # get on corrupt file should raise CorruptStorageFileError
        with pytest.raises(CorruptStorageFileError):
            store.get("corrupt_record")
