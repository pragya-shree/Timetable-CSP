"""Unit and integration tests for Bulk Timetable Setup and Importer.

Covers:
1. Pasted tabular data ingestion (TSV from Excel/Sheets, CSV, semicolon).
2. Header alias normalization.
3. CSV and XLSX file parsing.
4. Multi-sheet complete institutional workbook parsing.
5. Row-level validation: duplicate codes, missing required fields, invalid references.
6. Partial imports (valid rows accepted, invalid rejected with exact error messages).
7. Downloadable CSV and XLSX template generators.
8. Two-way availability matrix conversion (Grid DataFrame <-> slot coordinate strings).
"""

from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import pytest

from core.bulk_importer import (
    parse_pasted_tabular_text,
    parse_uploaded_file,
    parse_complete_workbook,
    validate_bulk_rows,
    generate_csv_template,
    generate_xlsx_template,
    generate_combined_workbook_template,
    teacher_slots_to_matrix,
    matrix_to_teacher_slots,
    normalize_column_name,
)
from core.custom_builder import validate_and_build_custom_dataset
from core.backtracking import solve_backtracking


class TestPastedTabularIngestion:
    def test_paste_tsv_subjects_with_headers(self):
        tsv = (
            "subject_id\tsubject_name\tclass_id\tteacher_id\ttype\tweekly_periods\tlab_duration\n"
            "CS101\tProgramming\tCSE_1\tT1\ttheory\t4\t1\n"
            "CS102\tDigital Logic\tCSE_1\tT2\ttheory\t3\t1\n"
        )
        df = parse_pasted_tabular_text(tsv, "subjects")
        assert len(df) == 2
        assert list(df["subject_id"]) == ["CS101", "CS102"]
        assert list(df["class_id"]) == ["CSE_1", "CSE_1"]

    def test_paste_tsv_with_alias_headers(self):
        """Header names copied from Excel like 'Code', 'Title', 'Hours' should be normalized."""
        tsv = (
            "Code\tTitle\tClass\tFaculty\tType\tHours\tDuration\n"
            "CS201\tData Structures\tCSE_2\tT1\ttheory\t4\t1\n"
        )
        df = parse_pasted_tabular_text(tsv, "subjects")
        assert len(df) == 1
        assert "subject_id" in df.columns
        assert "weekly_periods" in df.columns
        assert df.iloc[0]["subject_id"] == "CS201"
        assert df.iloc[0]["weekly_periods"] == "4"

    def test_paste_tsv_without_headers(self):
        """Raw rows copied from spreadsheets without header row."""
        tsv = (
            "CS301\tOperating Systems\tCSE_1\tT1\ttheory\t4\t1\n"
            "CS302\tComputer Networks\tCSE_1\tT2\ttheory\t4\t1\n"
        )
        df = parse_pasted_tabular_text(tsv, "subjects")
        assert len(df) == 2
        assert df.iloc[0]["subject_id"] == "CS301"
        assert df.iloc[1]["subject_name"] == "Computer Networks"

    def test_paste_semicolon_delimited(self):
        semi = "room_id;room_name;capacity;type\nR101;Hall A;60;classroom\n"
        df = parse_pasted_tabular_text(semi, "rooms")
        assert len(df) == 1
        assert df.iloc[0]["room_id"] == "R101"
        assert df.iloc[0]["capacity"] == "60"


class TestRowLevelValidationAndPartialImports:
    def test_subject_validation_accepts_valid_and_rejects_invalid_rows(self):
        df = pd.DataFrame([
            {"subject_id": "CS101", "subject_name": "Prog", "class_id": "CSE_1", "teacher_id": "T1", "type": "theory", "weekly_periods": "4", "lab_duration": "1"},
            {"subject_id": "", "subject_name": "No ID", "class_id": "CSE_1", "teacher_id": "T1", "type": "theory", "weekly_periods": "4", "lab_duration": "1"},
            {"subject_id": "CS102", "subject_name": "Invalid Class", "class_id": "NONEXISTENT", "teacher_id": "T1", "type": "theory", "weekly_periods": "4", "lab_duration": "1"},
            {"subject_id": "CS103", "subject_name": "Negative Hours", "class_id": "CSE_1", "teacher_id": "T1", "type": "theory", "weekly_periods": "-3", "lab_duration": "1"},
            {"subject_id": "CS101", "subject_name": "Duplicate ID", "class_id": "CSE_1", "teacher_id": "T1", "type": "theory", "weekly_periods": "4", "lab_duration": "1"},
        ])
        res = validate_bulk_rows(
            df,
            "subjects",
            known_classes={"CSE_1", "CSE_2"},
            known_teachers={"T1", "T2"},
        )
        assert not res.is_valid
        # Only 1 valid row accepted
        assert len(res.accepted_df) == 1
        assert res.accepted_df.iloc[0]["subject_id"] == "CS101"

        # 4 rejected rows with specific row errors
        assert len(res.rejected_rows) == 4
        assert res.rejected_rows[0].row_number == 2  # Missing ID
        assert res.rejected_rows[1].row_number == 3  # Unknown class
        assert res.rejected_rows[2].row_number == 4  # Negative hours
        assert res.rejected_rows[3].row_number == 5  # Duplicate ID

    def test_teacher_validation_duplicate_prevention(self):
        df = pd.DataFrame([
            {"teacher_id": "T1", "teacher_name": "Dr. Turing", "max_weekly_load": "14"},
            {"teacher_id": "T1", "teacher_name": "Duplicate Turing", "max_weekly_load": "10"},
        ])
        res = validate_bulk_rows(df, "teachers")
        assert not res.is_valid
        assert len(res.accepted_df) == 1
        assert len(res.rejected_rows) == 1
        assert "Duplicate teacher_id 'T1'" in res.rejected_rows[0].errors[0]

    def test_room_validation_capacity_and_type(self):
        df = pd.DataFrame([
            {"room_id": "R1", "room_name": "Valid Room", "capacity": "40", "type": "classroom"},
            {"room_id": "R2", "room_name": "Zero Cap", "capacity": "0", "type": "classroom"},
            {"room_id": "R3", "room_name": "Bad Type", "capacity": "30", "type": "auditorium"},
        ])
        res = validate_bulk_rows(df, "rooms")
        assert len(res.accepted_df) == 1
        assert len(res.rejected_rows) == 2


class TestTemplateGenerationAndFileImports:
    def test_csv_template_generation_and_roundtrip(self):
        csv_str = generate_csv_template("subjects")
        assert "subject_id" in csv_str
        assert "weekly_periods" in csv_str

        # Re-import generated template
        df = parse_pasted_tabular_text(csv_str, "subjects")
        assert len(df) >= 3
        res = validate_bulk_rows(df, "subjects")
        assert res.is_valid

    def test_xlsx_template_generation_and_parsing(self):
        xlsx_bytes = generate_xlsx_template("teachers")
        assert len(xlsx_bytes) > 0

        df = parse_uploaded_file(xlsx_bytes, "teachers_template.xlsx", "teachers")
        assert len(df) >= 3
        assert "teacher_id" in df.columns
        res = validate_bulk_rows(df, "teachers")
        assert res.is_valid

    def test_combined_multi_sheet_workbook_import(self):
        wb_bytes = generate_combined_workbook_template()
        custom_data, messages = parse_complete_workbook(wb_bytes)

        assert "config" in custom_data
        assert len(custom_data["classes"]) >= 2
        assert len(custom_data["teachers"]) >= 2
        assert len(custom_data["subjects"]) >= 3
        assert len(custom_data["rooms"]) >= 2

        # Validate that the template builds a 100% valid dataset
        val_res = validate_and_build_custom_dataset(
            config=custom_data["config"],
            classes_data=custom_data["classes"],
            teachers_data=custom_data["teachers"],
            subjects_data=custom_data["subjects"],
            rooms_data=custom_data["rooms"],
        )
        assert val_res.is_valid
        assert val_res.dataset is not None

        # Verify it solves cleanly with Backtracking
        solve_res = solve_backtracking(val_res.dataset, use_mrv=True, use_fc=True)
        assert solve_res.success
        assert solve_res.timetable is not None


class TestTeacherAvailabilityMatrix:
    def test_slots_to_matrix_and_roundtrip(self):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        periods = 6

        unavail_str = "Friday P6, Monday P1"
        pref_str = "Tuesday P2, Wednesday P3"

        matrix = teacher_slots_to_matrix(unavail_str, pref_str, days, periods)
        assert matrix.shape == (6, 5)

        # Check cell values
        assert "Unavailable" in matrix.loc["P6", "Friday"]
        assert "Unavailable" in matrix.loc["P1", "Monday"]
        assert "Preferred" in matrix.loc["P2", "Tuesday"]
        assert "Available" in matrix.loc["P3", "Monday"]

        # Convert back
        new_unavail, new_pref = matrix_to_teacher_slots(matrix, days)
        assert "Monday P1" in new_unavail
        assert "Friday P6" in new_unavail
        assert "Tuesday P2" in new_pref
        assert "Wednesday P3" in new_pref

    def test_default_matrix_is_all_available(self):
        days = ["Monday", "Tuesday"]
        matrix = teacher_slots_to_matrix("", "", days, 4)
        for col in days:
            for val in matrix[col]:
                assert "Available" in val
                assert "Unavailable" not in val


class TestTableManipulationHelpers:
    def test_append_blank_rows_classes(self):
        from core.bulk_importer import append_blank_rows
        df = pd.DataFrame([{"class_id": "C1", "class_name": "Class 1", "strength": 30, "lunch_slot": 4}])
        updated = append_blank_rows(df, "classes", count=3)
        assert len(updated) == 4
        assert updated.iloc[1]["class_id"] == "CLASS_2"

    def test_merge_imported_rows_deduplicates_by_id(self):
        from core.bulk_importer import merge_imported_rows
        existing = pd.DataFrame([
            {"room_id": "R1", "room_name": "Old Name", "capacity": 30, "type": "classroom"},
            {"room_id": "R2", "room_name": "Room 2", "capacity": 40, "type": "classroom"},
        ])
        incoming = pd.DataFrame([
            {"room_id": "R1", "room_name": "Updated Name", "capacity": 35, "type": "classroom"},
            {"room_id": "R3", "room_name": "Room 3", "capacity": 50, "type": "lab"},
        ])
        merged = merge_imported_rows(existing, incoming, "rooms", mode="append")
        assert len(merged) == 3
        # R1 should be updated
        r1_row = merged[merged["room_id"] == "R1"].iloc[0]
        assert r1_row["room_name"] == "Updated Name"
        assert int(r1_row["capacity"]) == 35

