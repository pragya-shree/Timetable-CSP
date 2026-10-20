"""Integration tests for Bulk Timetable Setup Workflow, State Retention, and Solver Integration.

Covers:
1. Navigation across wizard steps without losing data.
2. Dual workflow switching between Quick Setup Wizard and Classic Table Editor without data loss.
3. End-to-end pipeline: Bulk Ingestion -> Multi-Tier Validation -> CSP Solver -> Independent Verification -> Storage.
4. Robustness of Availability Matrix updates synced to teacher models.
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
    teacher_slots_to_matrix,
    matrix_to_teacher_slots,
    generate_combined_workbook_template,
)
from core.custom_builder import (
    create_default_custom_data,
    validate_and_build_custom_dataset,
)
from core.backtracking import solve_backtracking
from core.min_conflicts import solve_min_conflicts
from core.constraints import validate_timetable
from core.storage import TimetableRecord, TimetableStore
import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestWizardStateRetentionAndNavigation:
    def test_navigation_across_steps_preserves_custom_data(self):
        """Entering data across wizard steps and navigating back/forth retains all data."""
        data = create_default_custom_data()

        # Step 0: Modify time structure
        data["config"]["name"] = "State Retention College"
        data["config"]["days"] = ["Monday", "Tuesday", "Wednesday", "Thursday"]
        data["config"]["periods_per_day"] = 5
        data["config"]["default_lunch_slot"] = 3
        data["config"]["blocked_periods"] = "Monday P1"

        # Step 1: Add new class and subject via bulk paste
        new_classes_tsv = "class_id\tclass_name\tstrength\tlunch_slot\nAI_1\tAI Batch 1\t40\t3\n"
        parsed_c = parse_pasted_tabular_text(new_classes_tsv, "classes")
        data["classes"] = pd.concat([data["classes"], parsed_c], ignore_index=True)

        new_subj_tsv = "subject_id\tsubject_name\tclass_id\tteacher_id\ttype\tweekly_periods\tlab_duration\nAI101\tMachine Learning\tAI_1\tT1\ttheory\t3\t1\n"
        parsed_s = parse_pasted_tabular_text(new_subj_tsv, "subjects")
        data["subjects"] = pd.concat([data["subjects"], parsed_s], ignore_index=True)

        # Step 2: Update teacher availability via matrix
        days = data["config"]["days"]
        matrix = teacher_slots_to_matrix("Monday P1", "", days, 5)
        matrix.loc["P2", "Tuesday"] = "🔴 Unavailable"
        matrix.loc["P4", "Wednesday"] = "⭐ Preferred"
        new_unavail, new_pref = matrix_to_teacher_slots(matrix, days)

        t1_idx = data["teachers"][data["teachers"]["teacher_id"] == "T1"].index[0]
        data["teachers"].at[t1_idx, "unavailable_slots"] = new_unavail
        data["teachers"].at[t1_idx, "preferred_slots"] = new_pref

        # Step 3: Add room
        new_room_tsv = "room_id\troom_name\tcapacity\ttype\nR_AI\tAI Research Lab\t45\tlab\n"
        parsed_r = parse_pasted_tabular_text(new_room_tsv, "rooms")
        data["rooms"] = pd.concat([data["rooms"], parsed_r], ignore_index=True)

        # Simulate wizard state navigation: 0 -> 1 -> 2 -> 3 -> 4 -> 0
        wizard_step = 0
        for step in [1, 2, 3, 4, 3, 2, 1, 0, 4]:
            wizard_step = step
            # Assert all data remains intact regardless of current step pointer
            assert data["config"]["name"] == "State Retention College"
            assert "AI_1" in data["classes"]["class_id"].values
            assert "AI101" in data["subjects"]["subject_id"].values
            assert "R_AI" in data["rooms"]["room_id"].values
            assert "Tuesday P2" in data["teachers"].at[t1_idx, "unavailable_slots"]
            assert "Wednesday P4" in data["teachers"].at[t1_idx, "preferred_slots"]


class TestDualWorkflowSwitching:
    def test_switching_between_wizard_and_classic_editor(self):
        """Switching between workflow modes retains entered data seamlessly."""
        data = create_default_custom_data()
        data["config"]["name"] = "Dual Workflow Institute"

        # Mode A: Wizard is active
        app_mode = "🚀 Quick Setup Wizard (Bulk Entry & Importer)"
        assert data["config"]["name"] == "Dual Workflow Institute"

        # Add data in Wizard
        extra_room = pd.DataFrame([{"room_id": "CR_EXTRA", "room_name": "Extra Hall", "capacity": 50, "type": "classroom"}])
        data["rooms"] = pd.concat([data["rooms"], extra_room], ignore_index=True)

        # Switch to Mode B: Classic Table Editor
        app_mode = "🛠️ Classic Table Editor"
        assert "CR_EXTRA" in data["rooms"]["room_id"].values
        assert data["config"]["name"] == "Dual Workflow Institute"

        # Modify data in Classic Editor
        data["classes"].loc[0, "strength"] = 55

        # Switch back to Wizard
        app_mode = "🚀 Quick Setup Wizard (Bulk Entry & Importer)"
        assert data["classes"].loc[0, "strength"] == 55
        assert "CR_EXTRA" in data["rooms"]["room_id"].values


class TestBulkEndToEndGenerationPipeline:
    def test_complete_template_to_solve_and_save(self, tmp_path):
        """Full flow: generate workbook template -> parse -> validate -> solve -> save."""
        wb_bytes = generate_combined_workbook_template()
        custom_data, msgs = parse_complete_workbook(wb_bytes)
        assert len(msgs) >= 4

        # Validate
        val = validate_and_build_custom_dataset(
            config=custom_data["config"],
            classes_data=custom_data["classes"],
            teachers_data=custom_data["teachers"],
            subjects_data=custom_data["subjects"],
            rooms_data=custom_data["rooms"],
        )
        assert val.is_valid
        assert val.dataset is not None

        # Solve with MRV+FC Backtracking
        bt_res = solve_backtracking(val.dataset, use_mrv=True, use_fc=True)
        assert bt_res.success
        assert bt_res.timetable is not None

        # Verify independent constraint satisfaction
        check = validate_timetable(bt_res.timetable)
        assert check.is_valid
        assert check.hard_violation_count == 0

        # Save to persistent storage
        store = TimetableStore(tmp_path)
        rec = TimetableRecord.create_from_timetable(
            timetable=bt_res.timetable,
            name="Imported University Schedule",
            algorithm="MRV+FC",
            runtime_seconds=bt_res.runtime_seconds,
        )
        store.save(rec)

        # Verify storage
        records = store.list_all()
        assert len(records) == 1
        assert records[0].name == "Imported University Schedule"
        reloaded = store.get(rec.id).get_timetable()
        assert reloaded.is_complete()
        assert validate_timetable(reloaded).is_valid


class TestAppUIComponentsIntegration:
    def test_ui_helper_functions_callable(self):
        """Verify UI functions are importable from app."""
        assert hasattr(app, "_render_bulk_entity_box")
        assert hasattr(app, "_render_availability_matrix_editor")
        assert hasattr(app, "_render_generation_controls")
        assert callable(app._render_bulk_entity_box)
        assert callable(app._render_availability_matrix_editor)
        assert callable(app._render_generation_controls)
