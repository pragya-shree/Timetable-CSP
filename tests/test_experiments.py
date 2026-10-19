from __future__ import annotations
import csv, importlib.util, json, sys
from pathlib import Path
from typing import Any, Dict, List
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import load_dataset_from_file
from core.min_conflicts import solve_min_conflicts


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "run_experiments", PROJECT_ROOT / "experiments" / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_small():
    return load_dataset_from_file(PROJECT_ROOT / "datasets" / "small_dataset.json")


def _dummy(n=3):
    base = dict(config="plain", dataset="small", success=True, status="solved",
                runtime_seconds=0.5, iterations=10,
                conflicted_variable_selections=None, candidate_placements_evaluated=None,
                initial_hard_conflicts=None, final_hard_violations=0,
                total_cost=5, random_seed=None, cost_history=None,
                soft_s1_gaps=1, soft_s2_repeats=0, soft_s3_non_consecutive=2)
    return [dict(base, config=f"cfg_{i}") for i in range(n)]


# 1 - configuration
class TestExperimentConfiguration:
    def test_five_algorithms(self):
        assert set(_load_mod().SOLVER_CONFIGS) == {"plain","mrv","mrv_degree","mrv_degree_fc","min_conflicts"}

    def test_two_datasets(self):
        assert set(_load_mod().DATASET_PATHS) == {"small","large"}

    def test_dataset_files_exist(self):
        mod = _load_mod()
        for _, rel in mod.DATASET_PATHS.items():
            assert (PROJECT_ROOT / rel).exists()


# 2 - run_solver metrics
REQUIRED = {"config","dataset","success","status","runtime_seconds","total_cost"}

class TestRunSolverMetrics:
    def setup_method(self):
        self.mod = _load_mod()
        self.ds = _load_small()

    def test_plain_keys(self):         assert REQUIRED <= set(self.mod.run_solver("plain","small",self.ds))
    def test_mrv_keys(self):           assert REQUIRED <= set(self.mod.run_solver("mrv","small",self.ds))
    def test_mrv_degree_keys(self):    assert REQUIRED <= set(self.mod.run_solver("mrv_degree","small",self.ds))
    def test_mrv_degree_fc_keys(self): assert REQUIRED <= set(self.mod.run_solver("mrv_degree_fc","small",self.ds))

    def test_bt_mc_metric_null(self):
        assert self.mod.run_solver("plain","small",self.ds).get("conflicted_variable_selections") is None

    def test_bt_cost_history_null(self):
        assert self.mod.run_solver("plain","small",self.ds).get("cost_history") is None

    def test_runtime_positive(self):
        assert self.mod.run_solver("plain","small",self.ds)["runtime_seconds"] > 0


# 3 - cost history
class TestMinConflictsCostHistory:
    def setup_method(self): self.ds = _load_small()

    def test_enabled_returns_list(self):
        r = solve_min_conflicts(self.ds, max_iterations=200, record_cost_history=True, random_seed=42)
        assert r.cost_history is not None and isinstance(r.cost_history, list)

    def test_disabled_falsy(self):
        r = solve_min_conflicts(self.ds, max_iterations=200, record_cost_history=False, random_seed=42)
        assert not r.cost_history

    def test_items_2tuples(self):
        r = solve_min_conflicts(self.ds, max_iterations=200, record_cost_history=True, random_seed=42)
        for e in (r.cost_history or []):
            assert len(e) == 2
            assert isinstance(e[0], int) and isinstance(e[1], (int, float))

    def test_non_decreasing_iters(self):
        r = solve_min_conflicts(self.ds, max_iterations=200, record_cost_history=True, random_seed=42)
        its = [e[0] for e in (r.cost_history or [])]
        for i in range(1, len(its)):
            assert its[i] >= its[i-1]

    def test_genuine_per_iter(self):
        r = solve_min_conflicts(self.ds, max_iterations=50, record_cost_history=True,
                                random_seed=1234, time_limit_seconds=None)
        if r.iterations > 5 and r.cost_history:
            assert len(r.cost_history) > 2


# 4 - save_results
CSV_COLS = ["config","dataset","success","status","runtime_seconds","iterations","total_cost"]

class TestSaveResults:
    def setup_method(self): self.mod = _load_mod()

    def test_csv_created(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(3))
        assert (tmp_path / "experiment_runs.csv").exists()

    def test_csv_columns(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(3))
        with open(tmp_path / "experiment_runs.csv", newline="", encoding="utf-8") as f:
            headers = list(csv.DictReader(f).fieldnames or [])
        for c in CSV_COLS:
            assert c in headers

    def test_csv_row_count(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(5))
        with open(tmp_path / "experiment_runs.csv", newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 5

    def test_json_created(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(2))
        assert (tmp_path / "experiment_metadata.json").exists()

    def test_json_keys(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(2))
        meta = json.loads((tmp_path / "experiment_metadata.json").read_text())
        for k in ("timestamp","configurations","datasets","run_count"):
            assert k in meta

    def test_json_run_count(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(7))
        assert json.loads((tmp_path / "experiment_metadata.json").read_text())["run_count"] == 7

    def test_summary_created(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(_dummy(4))
        assert (tmp_path / "experiment_summary.csv").exists()

    def test_summary_success_rate_column(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path
        m = _dummy(4); m[0]["success"] = False
        self.mod.save_results(m)
        with open(tmp_path / "experiment_summary.csv", newline="", encoding="utf-8") as f:
            assert any("success_rate" in row for row in csv.DictReader(f))


# 5 - success rate
def _mc(s, t):
    return [dict(config="min_conflicts", dataset="small", success=i<s,
                 status="solved" if i<s else "limit_reached", runtime_seconds=1.0,
                 iterations=100, conflicted_variable_selections=100,
                 candidate_placements_evaluated=500, initial_hard_conflicts=10,
                 final_hard_violations=0 if i<s else 2, total_cost=5,
                 random_seed=i+1, cost_history=[(j, 10-j) for j in range(5)],
                 soft_s1_gaps=1, soft_s2_repeats=0, soft_s3_non_consecutive=0)
            for i in range(t)]

class TestSuccessRateCalculation:
    def setup_method(self): self.mod = _load_mod()

    def _sr(self, tmp_path, metrics):
        self.mod.RESULTS_DIR = tmp_path; self.mod.save_results(metrics)
        with open(tmp_path / "experiment_summary.csv", newline="", encoding="utf-8") as f:
            return float(list(csv.DictReader(f))[0]["success_rate"])

    def test_all_succeed(self, tmp_path):   assert self._sr(tmp_path, _mc(4,4))  == pytest.approx(1.0)
    def test_partial(self, tmp_path):        assert self._sr(tmp_path, _mc(3,10)) == pytest.approx(0.3)
    def test_all_fail(self, tmp_path):       assert self._sr(tmp_path, _mc(0,5))  == pytest.approx(0.0)


# 6 - plot
class TestPlotGeneration:
    def setup_method(self): self.mod = _load_mod()

    def _mh(self, n):
        return [dict(config="min_conflicts", dataset="small", success=True,
                     status="solved", runtime_seconds=0.5, iterations=20,
                     total_cost=5, random_seed=i+1,
                     cost_history=[(j, max(0, 15-j)) for j in range(1, 21)])
                for i in range(n)]

    def test_png_created(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path
        self.mod.plot_min_conflicts_cost(self._mh(2))
        assert (tmp_path / "min_conflicts_cost_vs_iterations.png").exists()

    def test_no_crash_no_mc_runs(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path
        self.mod.plot_min_conflicts_cost([{"config":"plain","dataset":"small","cost_history":None}])

    def test_no_crash_empty_history(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path
        self.mod.plot_min_conflicts_cost([{"config":"min_conflicts","dataset":"small",
                                           "success":False,"cost_history":[],"random_seed":42}])


# 7 - failed runs
class TestFailedRunHandling:
    def setup_method(self): self.mod = _load_mod()

    def test_failed_row_written(self, tmp_path):
        self.mod.RESULTS_DIR = tmp_path
        self.mod.save_results([dict(config="plain", dataset="large", success=False,
                                    status="limit_reached", runtime_seconds=60.0,
                                    iterations=None, conflicted_variable_selections=None,
                                    candidate_placements_evaluated=None, initial_hard_conflicts=None,
                                    final_hard_violations=5, total_cost=None, random_seed=None,
                                    cost_history=None, soft_s1_gaps=None,
                                    soft_s2_repeats=None, soft_s3_non_consecutive=None)])
        with open(tmp_path / "experiment_runs.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["success"] == "False"
        assert rows[0]["status"] == "limit_reached"


# 8 - seed recording
class TestRandomSeedRecording:
    def test_seed_propagated(self):
        r = solve_min_conflicts(_load_small(), max_iterations=50, random_seed=77777, time_limit_seconds=None)
        assert r.random_seed == 77777

    def test_none_seed_accepted(self):
        r = solve_min_conflicts(_load_small(), max_iterations=50, random_seed=None, time_limit_seconds=None)
        assert r.random_seed is None or isinstance(r.random_seed, int)
