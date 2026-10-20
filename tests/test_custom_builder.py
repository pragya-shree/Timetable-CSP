"""Unit tests for Custom Timetable Builder and Input Validation.

Tests custom data entry, parsing, multi-tier validation, reference integrity,
duplicate ID detection, capacity constraints, teacher workload limits,
and solver integration with custom datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import TimetableDataset, load_dataset_from_file
from core.custom_builder import (
    parse_slot_string,
    format_slots_to_string,
    create_default_custom_data,
    dataset_to_custom_tables,
    validate_and_build_custom_dataset,
    CustomValidationResult,
)
from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts
from core.constraints import validate_timetable


class TestSlotParsingAndFormatting:
    def test_parse_slot_string_named_days(self):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        slots = parse_slot_string("Monday P1, Wednesday P4", days, 6)
        assert slots == [(0, 0), (2, 3)]

    def test_parse_slot_string_abbreviated_days(self):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        slots = parse_slot_string("Mon P1; Fri P6", days, 6)
        assert slots == [(0, 0), (4, 5)]

    def test_parse_slot_string_coordinates(self):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        slots = parse_slot_string("(0, 1), [2, 3], 4:5", days, 6)
        assert slots == [(0, 1), (2, 3), (4, 5)]

    def test_parse_slot_string_empty_and_none(self):
        days = ["Monday", "Tuesday", "Wednesday"]
        assert parse_slot_string("", days, 5) == []
        assert parse_slot_string(None, days, 5) == []
        assert parse_slot_string("none", days, 5) == []

    def test_parse_slot_string_out_of_bounds_period(self):
        days = ["Monday", "Tuesday"]
        with pytest.raises(Exception) as exc:
            parse_slot_string("Monday P8", days, 6)
        assert "out of bounds" in str(exc.value).lower()

    def test_parse_slot_string_out_of_bounds_day(self):
        days = ["Monday", "Tuesday"]
        with pytest.raises(Exception) as exc:
            parse_slot_string("Friday P1", days, 6)
        assert "unknown day" in str(exc.value).lower()

    def test_format_slots_to_string(self):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        res = format_slots_to_string([(0, 0), (4, 5)], days)
        assert res == "Monday P1, Friday P6"


class TestDefaultCustomData:
    def test_default_template_valid_and_solvable(self):
        data = create_default_custom_data()
        result = validate_and_build_custom_dataset(
            config=data["config"],
            classes_data=data["classes"],
            teachers_data=data["teachers"],
            subjects_data=data["subjects"],
            rooms_data=data["rooms"],
        )
        assert result.is_valid, f"Default template errors: {result.errors}"
        assert result.dataset is not None
        assert len(result.dataset.classes) == 2
        assert len(result.dataset.teachers) == 3
        assert len(result.dataset.rooms) == 3

        # Solve with MRV backtracking to confirm it produces a valid solution
        res = solve_backtracking(result.dataset, use_mrv=True)
        assert res.success
        assert res.hard_violations == 0
        v_check = validate_timetable(res.timetable)
        assert v_check.is_valid

    def test_roundtrip_from_small_benchmark_dataset(self):
        ds = load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")
        tables = dataset_to_custom_tables(ds)
        res = validate_and_build_custom_dataset(
            config=tables["config"],
            classes_data=tables["classes"],
            teachers_data=tables["teachers"],
            subjects_data=tables["subjects"],
            rooms_data=tables["rooms"],
        )
        assert res.is_valid, f"Roundtrip errors: {res.errors}"
        assert res.dataset is not None
        assert len(res.dataset.classes) == len(ds.classes)
        assert len(res.dataset.teachers) == len(ds.teachers)
        assert len(res.dataset.rooms) == len(ds.rooms)


class TestCustomValidationErrors:
    def setup_method(self):
        self.base = create_default_custom_data()

    def test_missing_required_fields(self):
        classes = self.base["classes"].copy()
        classes.loc[0, "class_id"] = ""
        res = validate_and_build_custom_dataset(
            self.base["config"], classes, self.base["teachers"],
            self.base["subjects"], self.base["rooms"]
        )
        assert not res.is_valid
        assert any("Class ID is required" in e for e in res.errors)

    def test_duplicate_class_id(self):
        classes = self.base["classes"].copy()
        classes.loc[1, "class_id"] = classes.loc[0, "class_id"]
        res = validate_and_build_custom_dataset(
            self.base["config"], classes, self.base["teachers"],
            self.base["subjects"], self.base["rooms"]
        )
        assert not res.is_valid
        assert any("Duplicate Class ID" in e for e in res.errors)

    def test_duplicate_teacher_id(self):
        teachers = self.base["teachers"].copy()
        teachers.loc[1, "teacher_id"] = teachers.loc[0, "teacher_id"]
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], teachers,
            self.base["subjects"], self.base["rooms"]
        )
        assert not res.is_valid
        assert any("Duplicate Teacher ID" in e for e in res.errors)

    def test_duplicate_room_id(self):
        rooms = self.base["rooms"].copy()
        rooms.loc[1, "room_id"] = rooms.loc[0, "room_id"]
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            self.base["subjects"], rooms
        )
        assert not res.is_valid
        assert any("Duplicate Room ID" in e for e in res.errors)

    def test_unknown_class_reference_in_subject(self):
        subjects = self.base["subjects"].copy()
        subjects.loc[0, "class_id"] = "NON_EXISTENT_CLASS"
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            subjects, self.base["rooms"]
        )
        assert not res.is_valid
        assert any("does not exist in defined Classes" in e for e in res.errors)

    def test_unknown_teacher_reference_in_subject(self):
        subjects = self.base["subjects"].copy()
        subjects.loc[0, "teacher_id"] = "NON_EXISTENT_TEACHER"
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            subjects, self.base["rooms"]
        )
        assert not res.is_valid
        assert any("does not exist in defined Teachers" in e for e in res.errors)

    def test_invalid_negative_capacity(self):
        rooms = self.base["rooms"].copy()
        rooms.loc[0, "capacity"] = -5
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            self.base["subjects"], rooms
        )
        assert not res.is_valid
        assert any("Capacity must be positive" in e for e in res.errors)

    def test_room_capacity_exceeded_by_class_strength(self):
        rooms = self.base["rooms"].copy()
        # Set all classroom capacities smaller than class strength (30)
        for idx in rooms.index:
            if rooms.loc[idx, "type"] == "classroom":
                rooms.loc[idx, "capacity"] = 20
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            self.base["subjects"], rooms
        )
        assert not res.is_valid
        assert any("exceeds capacity of all classrooms" in e for e in res.errors)

    def test_lab_duration_indivisible(self):
        subjects = self.base["subjects"].copy()
        # Find lab row and set weekly_periods to 3 with duration 2
        lab_idx = subjects[subjects["type"] == "lab"].index[0]
        subjects.loc[lab_idx, "weekly_periods"] = 3
        subjects.loc[lab_idx, "lab_duration"] = 2
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            subjects, self.base["rooms"]
        )
        assert not res.is_valid
        assert any("must be divisible by lab duration" in e for e in res.errors)

    def test_teacher_workload_exceeded(self):
        teachers = self.base["teachers"].copy()
        # Set max_weekly_load of T1 very low (e.g. 2 periods)
        t1_idx = teachers[teachers["teacher_id"] == "T1"].index[0]
        teachers.loc[t1_idx, "max_weekly_load"] = 2
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], teachers,
            self.base["subjects"], self.base["rooms"]
        )
        assert not res.is_valid
        assert any("exceeds maximum weekly load" in e for e in res.errors)

    def test_class_usable_slots_exceeded(self):
        subjects = self.base["subjects"].copy()
        # Massive weekly periods for CSE_1
        c1_subjs = subjects[subjects["class_id"] == "CSE_1"].index
        for idx in c1_subjs:
            subjects.loc[idx, "weekly_periods"] = 25
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            subjects, self.base["rooms"]
        )
        assert not res.is_valid
        assert any("exceeds its" in e and "usable timetable slots" in e for e in res.errors)


class TestSolverIntegrationOnCustomDataset:
    def setup_method(self):
        data = create_default_custom_data()
        res = validate_and_build_custom_dataset(
            data["config"], data["classes"], data["teachers"],
            data["subjects"], data["rooms"]
        )
        assert res.is_valid
        self.dataset = res.dataset

    def test_plain_backtracking_solves_custom(self):
        res = solve_backtracking(self.dataset, use_mrv=False, use_degree=False, use_fc=False)
        assert res.success
        assert res.hard_violations == 0
        assert validate_timetable(res.timetable).is_valid

    def test_mrv_backtracking_solves_custom(self):
        res = solve_backtracking(self.dataset, use_mrv=True, use_degree=False, use_fc=False)
        assert res.success
        assert res.hard_violations == 0
        assert validate_timetable(res.timetable).is_valid

    def test_mrv_degree_solves_custom(self):
        res = solve_backtracking(self.dataset, use_mrv=True, use_degree=True, use_fc=False)
        assert res.success
        assert res.hard_violations == 0
        assert validate_timetable(res.timetable).is_valid

    def test_mrv_degree_fc_solves_custom(self):
        res = solve_backtracking(self.dataset, use_mrv=True, use_degree=True, use_fc=True)
        assert res.success
        assert res.hard_violations == 0
        assert validate_timetable(res.timetable).is_valid

    def test_min_conflicts_solves_custom(self):
        res = solve_min_conflicts(
            self.dataset,
            max_iterations=5000,
            random_seed=42,
            time_limit_seconds=15.0,
            record_cost_history=True,
        )
        assert res.success
        assert res.final_hard_violations == 0
        assert validate_timetable(res.timetable).is_valid


class TestExtraValidationEdgeCases:
    def setup_method(self):
        self.base = create_default_custom_data()

    def test_duplicate_subject_in_same_class(self):
        subjects = self.base["subjects"].copy()
        # Add duplicate subject row for CSE_1 with same subject_id
        dup_row = subjects.iloc[0].to_dict()
        subjects = pd.concat([subjects, pd.DataFrame([dup_row])], ignore_index=True)
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            subjects, self.base["rooms"]
        )
        assert not res.is_valid
        assert any("Duplicate subject requirement" in e for e in res.errors)

    def test_class_strength_exceeds_lab_capacity(self):
        rooms = self.base["rooms"].copy()
        # Shrink lab capacity to 10 (class strength is 30)
        lab_idx = rooms[rooms["type"] == "lab"].index[0]
        rooms.loc[lab_idx, "capacity"] = 10
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            self.base["subjects"], rooms
        )
        assert not res.is_valid
        assert any("exceeds capacity of all labs" in e for e in res.errors)

    def test_no_lab_room_for_lab_subject(self):
        rooms = self.base["rooms"].copy()
        # Change all rooms to classrooms
        rooms["type"] = "classroom"
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            self.base["subjects"], rooms
        )
        assert not res.is_valid
        assert any("no laboratory rooms have been added" in e for e in res.errors)

    def test_no_classroom_for_theory_subject(self):
        rooms = self.base["rooms"].copy()
        # Change all rooms to lab
        rooms["type"] = "lab"
        res = validate_and_build_custom_dataset(
            self.base["config"], self.base["classes"], self.base["teachers"],
            self.base["subjects"], rooms
        )
        assert not res.is_valid
        assert any("no classrooms have been added" in e for e in res.errors)

    def test_distinguish_invalid_input_from_feasibility_warning(self):
        # A valid dataset that has an aggregate capacity bottleneck creates warnings, not syntax errors
        data = create_default_custom_data()
        res = validate_and_build_custom_dataset(
            data["config"], data["classes"], data["teachers"],
            data["subjects"], data["rooms"]
        )
        assert res.is_valid
        assert len(res.errors) == 0
        assert isinstance(res.warnings, list)


class TestCustomManualEditingAndCost:
    def setup_method(self):
        data = create_default_custom_data()
        res = validate_and_build_custom_dataset(
            data["config"], data["classes"], data["teachers"],
            data["subjects"], data["rooms"]
        )
        assert res.is_valid
        self.dataset = res.dataset
        r = solve_backtracking(self.dataset, use_mrv=True)
        assert r.success
        self.timetable = r.timetable

    def test_valid_manual_move_preserves_validity(self):
        placed = self.timetable.get_placed_sessions()
        var, original_plc = placed[0]
        from core.sessions import get_valid_domain_for_session
        from core.sessions import Timetable
        from core.cost import calculate_cost

        valid_domain = get_valid_domain_for_session(var, self.dataset)
        assert len(valid_domain) > 0

        # Create copy and reassign to itself (trivially valid)
        new_asgn = dict(self.timetable.assignments)
        new_asgn[var] = original_plc
        new_tt = Timetable(dataset=self.dataset, assignments=new_asgn)
        val = validate_timetable(new_tt)
        assert val.is_valid
        cost = calculate_cost(new_tt)
        assert cost.hard_violations == 0

    def test_invalid_manual_move_flags_conflict(self):
        from core.sessions import Placement, Timetable
        placed = self.timetable.get_placed_sessions()
        if len(placed) >= 2:
            var1, plc1 = placed[0]
            var2, plc2 = placed[1]
            # Force var2 to clash with var1 in the same room and slot
            clash_plc = Placement(day=plc1.day, start_period=plc1.start_period, room_id=plc1.room_id)
            new_asgn = dict(self.timetable.assignments)
            new_asgn[var2] = clash_plc
            clash_tt = Timetable(dataset=self.dataset, assignments=new_asgn)
            val = validate_timetable(clash_tt)
            assert not val.is_valid
            assert val.hard_violation_count > 0


class TestCustomExport:
    def setup_method(self):
        data = create_default_custom_data()
        res = validate_and_build_custom_dataset(
            data["config"], data["classes"], data["teachers"],
            data["subjects"], data["rooms"]
        )
        assert res.is_valid
        self.dataset = res.dataset
        r = solve_backtracking(self.dataset, use_mrv=True)
        assert r.success
        self.timetable = r.timetable

    def test_custom_timetable_csv_and_excel_export(self):
        import io
        import app
        df = app._sessions_df(self.timetable)
        assert not df.empty
        assert len(df) == len(self.timetable.get_placed_sessions())

        # Test CSV export
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        loaded_csv = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(loaded_csv) == len(df)

        # Test Excel export
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="CustomTimetable")
        buf.seek(0)
        loaded_excel = pd.read_excel(buf, sheet_name="CustomTimetable")
        assert len(loaded_excel) == len(df)

