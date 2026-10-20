# AI-Based Timetable Scheduler

**Course**: Artificial Intelligence (B.Tech, Year 2)  
**System**: Complete Constraint Satisfaction Problem (CSP) & Heuristic Local Search Scheduling Engine  
**Stack**: Python 3.10+, `pytest`, `streamlit`, `pandas`, `matplotlib`, `openpyxl`  

---

## 1. Project Purpose & System Overview

Academic timetable scheduling is a challenging combinatorial optimization problem: lectures and laboratory sessions across multiple student batches, faculty members, and physical venues must be placed on a weekly grid without collisions while adhering to institutional constraints and minimizing idle gaps.

This project delivers an end-to-end AI-based timetable scheduler with:
- **Strongly-Typed Data Models**: Dataclasses for time structures, subjects, teachers, class batches, rooms, and datasets.
- **Atomic CSP Variable Formulation**: Teaching sessions (`SessionVar`) where multi-period laboratories (e.g. 2-period lab) are atomic decision variables with `duration = 2`, preventing constraint thrashing.
- **Nine Hard Constraint Validators (H1–H9 + STRUCT)**: Rigorous validation with exact counting rules and zero tolerance for schedule invalidity.
- **Three Soft Penalty Evaluators (S1–S3) & Cost Function**: Quantitative metrics penalizing class day gaps (S1), same-day subject repeats (S2), and non-consecutive lab sessions (S3).
- **Five Search Algorithms**:
  1. Plain Backtracking (DFS baseline)
  2. Backtracking + MRV (Minimum Remaining Values)
  3. Backtracking + MRV + Degree Heuristic
  4. Backtracking + MRV + Degree + Forward Checking (active domain lookahead)
  5. Min-Conflicts Heuristic Local Search (with random walk and stagnation escaping)
- **Standardized Academic Benchmarks**: Small Dataset (3 classes, 6 teachers, 4 rooms) and Large Dataset (8 classes, 15 teachers, 8 rooms) with mathematical feasibility proofs.
- **Unified 6-Section Sidebar Navigation**:
  1. **Dashboard**: Central status overview, active timetable card, system workflow guide, and saved timetable history.
  2. **Create Timetable**: Full institutional configuration builder, multi-tier validation, solver execution, and post-generation actions.
  3. **My Timetables**: Local JSON-based persistent timetable history system (save, open, rename, duplicate, delete).
  4. **View & Edit Timetable**: Single authoritative schedule viewer (Class, Teacher, Room grids) and interactive manual editor with real-time conflict detection.
  5. **Experiments & Benchmarks**: Isolated experimental suite evaluating all 5 algorithms and recording genuine metrics without altering user timetables.
  6. **Export**: Synchronized export of the active schedule to CSV, Excel (`.xlsx`), and JSON.
- **Independent Solution Verification**: Every generated schedule is verified against all constraints using `validate_timetable` before being accepted as valid.
- **306+ Automated Unit and Integration Tests**: 100% clean test suite with zero failures and zero warnings.

---

## 2. CSP Formulation

### 2.1 Variables ($X$) — `SessionVar`
Each variable represents an atomic lecture or laboratory session that must be placed on the schedule grid:
$$\text{SessionVar} = (\text{session ID}, \text{class ID}, \text{subject ID}, \text{session no}, \text{teacher ID}, \text{duration}, \text{session type})$$

- **Theory Session**: Duration = 1 period. A subject with 4 weekly hours generates 4 distinct `SessionVar` instances (`session_no` $\in \{1, 2, 3, 4\}$).
- **Lab Session**: Duration = `lab_duration` (e.g. 2 consecutive periods). A lab with 2 weekly hours generates **one single session** occupying 2 consecutive periods. It is NOT modeled as two disconnected 1-hour sessions.

### 2.2 Domain Values ($D$) — `Placement`
A placement value represents a specific coordinate in space and time:
$$\text{Placement} = (\text{day}, \text{start period}, \text{room ID})$$
- $\text{day} \in \{0, \dots, \text{days per week} - 1\}$
- $\text{start period} \in \{0, \dots, \text{periods per day} - \text{duration}\}$
- A session occupies periods $\{p, p + 1, \dots, p + \text{duration} - 1\}$ on that day.
- $\text{room ID} \in \text{rooms}$

### 2.3 Assignment & Timetable
A timetable assignment is a mapping $\mathcal{A}: \text{SessionVar} \to \text{Placement} \cup \{\text{None}\}$.
- `is_complete()`: Returns `True` iff every required variable has a non-`None` placement.
- `is_valid()`: Returns `True` iff `is_complete() == True` AND `hard_violation_count == 0`. An empty or partial timetable is **never** valid.

---

## 3. Hard Constraints (H1–H9 + STRUCT)

Hard constraints represent physical and institutional impossibilities. A timetable with any hard violation is invalid.

| Constraint | Name | Exact Counting Rule | Viva Explanation |
|---|---|---|---|
| **H1** | Teacher Clash | **1 violation per conflicting pair of sessions per overlapping period**. If Teacher T is scheduled in 2 sessions simultaneously for 1 period = 1 violation. If a 2-period lab overlaps with another session on both periods = 2 violations. | Binary constraint. A teacher cannot physically be in two classrooms at the same time. |
| **H2** | Class Clash | **1 violation per conflicting pair of sessions for the same class per overlapping period**. | Binary constraint. A student batch cannot attend two lectures at the same time. |
| **H3** | Room Clash | **1 violation per conflicting pair of sessions in the same room per overlapping period**. | Binary constraint. A room cannot host two distinct sessions simultaneously. |
| **H4** | Room Capacity | **1 violation per placed session** where `room.capacity < class.strength`. | Unary constraint. Overcrowded classrooms cannot seat all enrolled students. |
| **H5** | Room Type | **1 violation per placed session** in an incompatible room (`lab` in classroom, or `theory` in lab). | Unary constraint. Labs require specialized equipment; theory lectures require standard lecture halls. |
| **H6** | Teacher Availability & Blocked Slots | **1 violation per occupied slot** falling on a teacher's `unavailable_slots` OR a global `blocked_periods` (e.g. college assembly). | Unary constraint on time coordinates. |
| **H7** | Weekly Syllabus Hours | **$\mid\text{assigned periods} - \text{required periods}\mid$ per (class, subject)**. Missing and extra periods both count. Unassigned sessions contribute directly as missing periods. | Quota constraint ensuring complete syllabus coverage. An empty timetable has violations equal to total required periods. |
| **H8** | Teacher Max Weekly Load | **$\max(0, \text{assigned periods} - \text{max weekly load})$** per teacher. | Workload limit protecting faculty from overwork (a 2-period lab adds 2 periods towards load). |
| **H9** | Class Lunch Break | **1 violation per occupied slot** falling on `class.lunch_slot`. | Institutional policy ensuring students have a daily dedicated lunch break. |
| **STRUCT** | Structural Validity | **1 violation per occurrence**: out-of-bounds day/period, multi-period session overflowing the end of the day, unknown entity ID, or teacher assigned to a subject they are not qualified to teach. | Ensures data integrity and prevents silent out-of-bounds corruption. |

---

## 4. Soft Constraints & Objective Cost Function

Soft constraints represent quality preferences. They do not render a timetable invalid, but higher penalties indicate poorer scheduling quality.

### 4.1 Implemented Soft Constraints (S1–S3)
- **S1: Minimize Gaps in Class Day**
  - Identifies idle waiting periods between the first and last occupied lecture for each class on each day.
  - *Exclusions*: The class's designated `lunch_slot` and any global `blocked_periods` are not counted as gaps.
  - Penalty: $\text{idle gap periods} \times w_{s1}$.
- **S2: Spread Subject Lectures Across Days**
  - Encourages spaced learning by penalizing multiple theory lectures of the same subject on the same day.
  - If class $C$ has $k > 1$ theory lectures of subject $S$ on day $D$, penalty is $(k - 1) \times w_{s2}$.
  - *Exemption*: Lab sessions are strictly exempt (a 2-period lab is 1 session and is not a duplicate).
- **S3: Consecutive Lab Periods**
  - By construction, an atomic lab session occupies consecutive slots $\{p, p + 1\}$.
  - S3 explicitly flags and penalizes:
    1. A lab placement that is interrupted by a class lunch break or a global blocked period.
    2. Multiple lab sessions of the same subject on the same day placed with idle periods between them.
  - Penalty: $\text{interrupted or separated slots} \times w_{s3}$.

### 4.2 Objective Cost Function
$$\text{Total Cost} = 1000 \times \text{hard violations} + (w_{s1} \cdot S_1 + w_{s2} \cdot S_2 + w_{s3} \cdot S_3)$$
- Default weights: $w_{s1} = 1, w_{s2} = 1, w_{s3} = 1$.
- Hard violations dominate the cost function ($1000\times$), ensuring that any feasible timetable is strictly preferred over an infeasible one.

---

## 5. Unified Navigation & Workflows

The application interface is structured into six synchronized sections:

### 5.1 Dashboard
- **Active Timetable Card**: Displays schedule name, algorithm, conflict status, total cost, and placed session count.
- **Quick Benchmark Solvers**: One-click solve buttons for Small and Large datasets.
- **Recent Saved Timetables**: Shows recently saved schedules with one-click activation.

### 5.2 Create Timetable (Integrated Dataset + Solver Workflow)
Combines dataset definition, validation, solver execution, and post-generation actions into a single workflow:
1. **Dataset Selection**:
   - Predefined Benchmark Dataset (Small or Large).
   - Custom Dataset Builder with interactive tables (`st.data_editor`) for Time Grid, Classes, Teachers, Subjects, and Rooms.
2. **Multi-Tier Validation**:
   - Required fields, positive values, duplicate IDs, cross-entity references, workload caps, and room capacities.
   - Actionable error messages and feasibility warnings.
3. **Integrated Solver Execution**:
   - Select algorithm (Plain, MRV, MRV+Degree, MRV+Degree+FC, Min-Conflicts).
   - Configure solver limits (nodes, iterations, time limits, seeds).
   - Runs solver and independently verifies returned timetable with `validate_timetable`.
4. **Post-Generation Actions**:
   - **View Timetable**: Activates the timetable and opens it in View & Edit Timetable.
   - **Save Timetable**: Saves the schedule with a user-given name to My Timetables.
   - **Regenerate**: Allows running another algorithm on the same dataset without losing previous work.

### 5.3 My Timetables (Persistent Local Storage)
- **Local Storage Architecture**: Timetables are stored locally as JSON files on the machine in `saved_timetables/` (e.g. `d:\Timetable\saved_timetables\tt_*.json`).
- **Record Attributes**:
  - Unique timetable ID
  - User-editable name
  - Creation and updated ISO 8601 timestamps
  - Version / revision number
  - Full dataset specification
  - Generated placements mapping
  - Algorithm used, runtime, and search metrics
  - Validation outcome and cost breakdown
- **Atomic Operations**:
  - **Open / Set Active**: Loads timetable into active state.
  - **Rename**: Edits display name and increments revision number.
  - **Duplicate**: Clones timetable under a new unique ID.
  - **Delete**: Safely removes record after user confirmation.
  - **JSON Backup**: Downloads single timetable backup file.
- **Fault Tolerance**: Atomic write-and-replace (`.tmp` to `.json`) prevents file corruption; malformed files are skipped gracefully during directory indexing.

### 5.4 View & Edit Timetable
- **Single Source of Truth**: Strictly reads from `st.session_state["active_timetable"]`.
- **Three Synchronized Views**:
  - **Class View**: Displays weekly schedule for each batch with lunch breaks and lab continuation markers `... (cont.)`.
  - **Teacher View**: Shows each faculty member's teaching commitments.
  - **Room View**: Shows venue occupancy.
- **Pandas 3.x Compatibility**: Implements backward- and forward-compatible styling via `_style_df` (using `Styler.map` on Pandas 3.x and `Styler.applymap` on Pandas 2.x), eliminating `AttributeError`.
- **Manual Schedule Editor**:
  - Pick any session and move it to a new Day, Start Period, and Room.
  - Immediately validates with `validate_timetable` and `calculate_cost`.
  - Clearly flags any hard violations introduced (labeled as "Edited Schedule with Conflicts").
  - **Save Changes**: Persists edits to the active timetable record in storage.
  - **Revert Changes**: Restores the last clean saved version.

### 5.5 Experiments & Benchmarks
- Benchmark evaluation suite for all 5 algorithms on Small and Large datasets.
- Evaluates 10 randomized Min-Conflicts trials per dataset.
- Generates `min_conflicts_cost_vs_iterations.png` convergence plot.
- **State Isolation**: Running or inspecting benchmark experiments never mutates or overwrites the user's active personal timetable.

### 5.6 Export
- Exports the active timetable's actual schedule to **CSV** (`timetable.csv`) and **Excel** (`timetable.xlsx` via `openpyxl`).
- Exports dataset to JSON (`dataset.json`) and full timetable backup (`timetable_backup.json`).

---

## 6. Solver Algorithms

### 6.1 Systematic Backtracking Solvers (`core/backtracking.py`)

1. **Plain Backtracking (Baseline)**: Systematic Depth-First Search in static variable generation order with dynamic collision checks.
2. **Backtracking + MRV (Minimum Remaining Values)**: Dynamically selects the variable with the fewest remaining legal placements.
3. **Backtracking + MRV + Degree Heuristic**: Selects variables with minimum legal domain size, breaking ties using dynamic constraint degree.
4. **Backtracking + MRV + Degree + Forward Checking (`mrv_degree_fc`)**: Maintains active domains in `current_domains`, prunes conflicting values 1 step ahead upon each assignment, and backtracks immediately on domain wipeout.

### 6.2 Min-Conflicts Local Search (`core/min_conflicts.py`)

- **Heuristic Repair Paradigm**: Starts from a complete initial placement drawn from valid unary domains and iteratively repairs the variable with the highest conflict count.
- **Atomic Conflict Scoring**: Evaluates exact binary clashes across H1 (Teacher), H2 (Class), and H3 (Room) with self-exclusion.
- **Plateau Escaping**: Random walk probability $p_{\text{walk}} = 0.05$, stagnation detection with perturbation after 150 non-improving steps, and non-stationary tie-breaking.

---

## 7. Benchmark Results

Measurements on Python 3.14 across identical datasets:

### 7.1 Systematic Backtracking Benchmark

| Dataset | Algorithm | Status | Runtime | Nodes Explored | Assignments Tried | Backtracks | Hard Violations | Soft Penalties (S1 / S2 / S3) | Total Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small Dataset** | Plain Backtracking | `SOLVED` | **0.0008s** | 40 | 642 | 0 | 0 | 1 / 25 / 0 | 26 |
| **Small Dataset** | Backtracking + MRV | `SOLVED` | **0.0177s** | 40 | 753 | 0 | 0 | 0 / 21 / 0 | **21** |
| **Small Dataset** | Backtracking + MRV + Degree | `SOLVED` | **0.0180s** | 40 | 753 | 0 | 0 | 0 / 21 / 0 | **21** |
| **Small Dataset** | Backtracking + MRV + Degree + FC | `SOLVED` | **0.0051s** | 40 | **39** | 0 | 0 | 0 / 21 / 0 | **21** |
| **Large Dataset** | Plain Backtracking | `SOLVED` | **0.0046s** | 105 | 4,029 | 0 | 0 | 4 / 46 / 0 | 50 |
| **Large Dataset** | Backtracking + MRV | `SOLVED` | **0.2994s** | 105 | 3,874 | 0 | 0 | 4 / 48 / 0 | 52 |
| **Large Dataset** | Backtracking + MRV + Degree | `SOLVED` | **0.3271s** | 105 | 3,874 | 0 | 0 | 4 / 48 / 0 | 52 |
| **Large Dataset** | Backtracking + MRV + Degree + FC | `SOLVED` | **0.0670s** | 105 | **104** | 0 | 0 | 4 / 48 / 0 | 52 |

### 7.2 Min-Conflicts Local Search Benchmark (10 Randomized Runs per Dataset)

| Dataset | Trials | Success Rate | Avg Runtime | Avg Iterations | Avg Hard Violations | Avg Soft Penalties (S1 / S2 / S3) | Avg Total Cost |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small Dataset** | 10 | **100%** | **0.0014s** | 16.3 | 0 | 9.8 / 8.8 / 0.0 | **18.6** |
| **Large Dataset** | 10 | **100%** | **0.0078s** | 44.5 | 0 | 38.7 / 16.5 / 0.0 | **55.2** |

---

## 8. Installation & Usage

### 8.1 Setup
Requires Python 3.10+:

```bash
cd D:\Timetable

# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Launching the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501`. Navigate through the sidebar:
1. `📊 Dashboard`: Status overview, active schedule details, quick benchmark loaders.
2. `🛠️ Create Timetable`: Define or select dataset, validate, and generate schedule.
3. `📁 My Timetables`: Manage saved schedules (open, rename, duplicate, delete).
4. `👁️ View & Edit Timetable`: Grid views (Class, Teacher, Room) and manual editor.
5. `🔬 Experiments & Benchmarks`: Review benchmark metrics and cost convergence plot.
6. `📥 Export`: Export schedule as CSV, Excel, or JSON.

### 8.3 Running Automated Experiments
```bash
python experiments/run_experiments.py
```

### 8.4 Running the Test Suite
```bash
python -m pytest -q
```
Current execution result:
```
........................................................................ [ 28%]
........................................................................ [ 57%]
........................................................................ [ 85%]
....................................                                     [100%]
252 passed in 6.47s
```

---

## 9. Project Structure

```
Timetable/
├── core/
│   ├── __init__.py          # Package metadata & solver exports
│   ├── models.py            # Dataclasses, JSON loading, schema & feasibility checks
│   ├── sessions.py          # SessionVar, Placement, Timetable container
│   ├── constraints.py       # H1-H9 + STRUCT validators, validate_timetable
│   ├── cost.py              # S1-S3 soft penalties, CostBreakdown, calculate_cost
│   ├── backtracking.py      # Plain, MRV, MRV+Degree, MRV+Degree+FC solvers
│   ├── min_conflicts.py     # Min-Conflicts Local Search solver
│   ├── custom_builder.py    # Custom timetable parsing, multi-tier validation, dataset builder
│   └── storage.py           # Persistent atomic JSON storage for My Timetables
├── datasets/
│   ├── small_dataset.json   # 3 classes, 6 teachers, 4 rooms, 5x6 grid
│   ├── large_dataset.json   # 8 classes, 15 teachers, 8 rooms, 5x7 grid
│   ├── small_feasible_schedule.json # Proof schedule
│   └── large_feasible_schedule.json # Proof schedule
├── experiments/
│   ├── run_experiments.py   # Benchmark runner
│   └── results/             # CSV, JSON metadata, and PNG convergence plot
├── saved_timetables/        # Persistent local JSON storage for user schedules
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_models.py       # Model validation tests
│   ├── test_sessions.py     # Session generation and placement tests
│   ├── test_constraints.py  # Isolated H1-H9 violation tests
│   ├── test_cost.py         # S1-S3 penalty tests
│   ├── test_datasets.py     # Dataset structure tests
│   ├── test_feasibility.py  # Proof schedule verification tests
│   ├── test_backtracking.py # Backtracking solver tests (51 tests)
│   ├── test_min_conflicts.py# Min-Conflicts solver tests (18 tests)
│   ├── test_experiments.py  # Experiment runner tests (32 tests)
│   ├── test_app.py          # Streamlit app tests (55 tests)
│   ├── test_custom_builder.py # Custom builder tests (33 tests)
│   ├── test_storage.py      # Timetable store tests (7 tests)
│   └── test_refactored_workflow.py # Workflow and regression tests (11 tests)
├── app.py                   # Refactored Streamlit application
├── run_solver.py            # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md
```
