"""Persistent Storage and History Management for Timetables.

Provides an atomic, JSON-based local storage system for saving, retrieving,
renaming, duplicating, and deleting generated and edited timetables.
Reconstructs model objects (TimetableDataset, Timetable) safely on load.
"""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models import TimetableDataset, load_dataset_from_dict
from core.sessions import Timetable
from core.constraints import validate_timetable
from core.cost import calculate_cost

_DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent / "saved_timetables"


class StorageError(Exception):
    """Base exception for timetable storage errors."""
    pass


class TimetableNotFoundError(StorageError):
    """Raised when a requested timetable record does not exist."""
    pass


class CorruptStorageFileError(StorageError):
    """Raised when a storage JSON file is malformed or missing essential keys."""
    pass


@dataclass
class TimetableRecord:
    """Serializable record storing a complete timetable and its metadata."""
    id: str
    name: str
    created_at: str
    updated_at: str
    version: int
    algorithm: str
    runtime_seconds: float | None
    search_metrics: dict[str, Any]
    validation: dict[str, Any]
    cost: dict[str, Any]
    dataset_dict: dict[str, Any]
    schedule_dict: dict[str, Any]

    def get_dataset(self) -> TimetableDataset:
        """Reconstructs and validates the TimetableDataset object."""
        return load_dataset_from_dict(self.dataset_dict)

    def get_timetable(self) -> Timetable:
        """Reconstructs the full Timetable object with its assigned placements."""
        ds = self.get_dataset()
        return Timetable.from_dict(self.schedule_dict, ds)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the record to a dictionary suitable for JSON dumping."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "algorithm": self.algorithm,
            "runtime_seconds": self.runtime_seconds,
            "search_metrics": self.search_metrics,
            "validation": self.validation,
            "cost": self.cost,
            "dataset": self.dataset_dict,
            "schedule": self.schedule_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimetableRecord:
        """Constructs a TimetableRecord from a raw dictionary with validation."""
        required_keys = ("id", "name", "created_at", "updated_at", "dataset", "schedule")
        for k in required_keys:
            if k not in data:
                raise CorruptStorageFileError(f"Missing required key '{k}' in timetable record.")

        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            version=int(data.get("version", 1)),
            algorithm=str(data.get("algorithm", "Unknown")),
            runtime_seconds=float(data["runtime_seconds"]) if data.get("runtime_seconds") is not None else None,
            search_metrics=dict(data.get("search_metrics", {})),
            validation=dict(data.get("validation", {})),
            cost=dict(data.get("cost", {})),
            dataset_dict=dict(data["dataset"]),
            schedule_dict=dict(data["schedule"]),
        )

    @classmethod
    def create_from_timetable(
        cls,
        timetable: Timetable,
        name: str,
        algorithm: str = "Manual",
        timetable_id: str | None = None,
        runtime_seconds: float | None = None,
        search_metrics: dict[str, Any] | None = None,
        version: int = 1,
    ) -> TimetableRecord:
        """Factory method to create a fresh record from an active Timetable object."""
        now_str = datetime.now(timezone.utc).isoformat()
        tid = timetable_id or f"tt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Run independent validation
        val_res = validate_timetable(timetable)
        val_data = {
            "is_valid": val_res.is_valid,
            "is_complete": val_res.is_complete,
            "hard_violation_count": val_res.hard_violation_count,
            "violations": [
                {
                    "constraint_id": v.constraint_id,
                    "message": v.message,
                    "sessions_involved": v.sessions_involved,
                    "details": v.details,
                }
                for v in val_res.violations
            ],
        }

        # Calculate cost
        cost_res = calculate_cost(timetable)
        cost_data = {
            "total_cost": cost_res.total_cost,
            "hard_violations": cost_res.hard_violations,
            "soft_penalties": {
                "s1_gaps": cost_res.s1_gaps,
                "s2_repeats": cost_res.s2_repeats,
                "s3_non_consecutive": cost_res.s3_non_consecutive,
            },
        }

        return cls(
            id=tid,
            name=name.strip() or "Untitled Timetable",
            created_at=now_str,
            updated_at=now_str,
            version=version,
            algorithm=algorithm,
            runtime_seconds=runtime_seconds,
            search_metrics=search_metrics or {},
            validation=val_data,
            cost=cost_data,
            dataset_dict=timetable.dataset.to_dict(),
            schedule_dict=timetable.to_dict(),
        )


class TimetableStore:
    """Manages persistent timetable storage in a designated directory."""

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, timetable_id: str) -> Path:
        clean_id = "".join(c for c in timetable_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{clean_id}.json"

    def save(self, record: TimetableRecord) -> TimetableRecord:
        """Atomically saves a timetable record to disk."""
        target_path = self._file_path(record.id)
        temp_path = self.storage_dir / f"{target_path.stem}.tmp_{uuid.uuid4().hex[:6]}"

        data = record.to_dict()
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            # Atomic rename / replace
            os.replace(temp_path, target_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

        return record

    def get(self, timetable_id: str) -> TimetableRecord:
        """Retrieves and parses a timetable record by ID."""
        path = self._file_path(timetable_id)
        if not path.exists():
            raise TimetableNotFoundError(f"Timetable '{timetable_id}' not found.")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TimetableRecord.from_dict(data)
        except (json.JSONDecodeError, CorruptStorageFileError) as exc:
            raise CorruptStorageFileError(f"Corrupt timetable file '{path.name}': {exc}") from exc

    def list_all(self) -> list[TimetableRecord]:
        """Lists all valid timetable records, sorted by updated_at descending.
        
        Corrupt or unreadable files are skipped gracefully.
        """
        records: list[TimetableRecord] = []
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rec = TimetableRecord.from_dict(data)
                records.append(rec)
            except Exception:
                # Gracefully skip corrupt or malformed files
                continue

        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    def delete(self, timetable_id: str) -> bool:
        """Deletes a timetable record file. Returns True if deleted, False if not found."""
        path = self._file_path(timetable_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def rename(self, timetable_id: str, new_name: str) -> TimetableRecord:
        """Renames a saved timetable and updates its version and timestamp."""
        record = self.get(timetable_id)
        record.name = new_name.strip() or record.name
        record.updated_at = datetime.now(timezone.utc).isoformat()
        record.version += 1
        return self.save(record)

    def duplicate(self, timetable_id: str, new_name: str | None = None) -> TimetableRecord:
        """Duplicates an existing timetable under a new unique ID."""
        record = self.get(timetable_id)
        new_id = f"tt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now_str = datetime.now(timezone.utc).isoformat()

        copied_record = TimetableRecord(
            id=new_id,
            name=new_name.strip() if new_name else f"{record.name} (Copy)",
            created_at=now_str,
            updated_at=now_str,
            version=1,
            algorithm=record.algorithm,
            runtime_seconds=record.runtime_seconds,
            search_metrics=dict(record.search_metrics),
            validation=dict(record.validation),
            cost=dict(record.cost),
            dataset_dict=dict(record.dataset_dict),
            schedule_dict=dict(record.schedule_dict),
        )
        return self.save(copied_record)
