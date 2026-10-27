# AI-Based Timetable Scheduler

**Course**: Artificial Intelligence (B.Tech, Year 2)  
**System**: Complete Constraint Satisfaction Problem (CSP) & Heuristic Local Search Scheduling Engine  
**Stack**: Python 3.10+, `pytest`, `streamlit`, `pandas`, `matplotlib`, `openpyxl`  

---

## 1. Project Purpose & System Overview

Academic timetable scheduling is a challenging combinatorial optimization problem: lectures and laboratory sessions across multiple student batches, faculty members, and physical venues must be placed on a weekly grid without collisions while adhering to institutional constraints and optimizing timetable quality.

This project delivers an end-to-end AI-based timetable scheduler featuring:
- **Strongly-Typed Data Models**: Dataclasses for time structures, subjects, teachers, class batches, rooms, and datasets.
- **Atomic CSP Variable Formulation**: Teaching sessions (`SessionVar`) where multi-period laboratories (e.g., 2-period lab) are atomic decision variables with `duration = 2`, preventing constraint thrashing.
- **Nine Hard Constraint Validators (H1–H9 + STRUCT)**: Rigorous validation with exact counting rules and zero tolerance for schedule invalidity.
- **Five Soft Penalty Evaluators (S1–S5) & Multi-Factor Cost Function**: Quantitative metrics penalizing class day gaps (S1), same-day subject repeats (S2), non-consecutive lab sessions (S3), placements outside teacher preferred slots (S4), and consecutive teacher lectures $>2$ in a row (S5).
- **Six Search and Optimization Algorithms**:
  1. Plain Backtracking (DFS baseline)
  2. Backtracking + MRV (Minimum Remaining Values)
  3. Backtracking + MRV + Degree Heuristic
  4. Backtracking + MRV + Degree + Forward Checking (active domain lookahead)
  5. Min-Conflicts Heuristic Local Search (with random walk, plateau escaping, and optional two-stage soft refinement)
  6. Hybrid CSP + Simulated Annealing (MRV+Degree+FC initial feasible solution + Metropolis soft-cost optimization)
- **Standardized Academic Benchmarks**: Small Dataset (3 classes, 6 teachers, 4 rooms), Medium Tight Dataset (4 classes, 4 teachers, 3 rooms with high contention), and Large Dataset (8 classes, 15 teachers, 8 rooms) with mathematical feasibility proofs.
- **Unified 6-Section Sidebar Navigation**:
  1. **Dashboard**: Central status overview, active timetable card, system workflow guide, and saved timetable history.
  2. **Create Timetable**: Dual input workflows (Quick Setup Wizard with Excel/Sheets TSV paste, CSV/XLSX imports, downloadable templates, interactive Teacher Availability Matrix, and Classic Table Editor), multi-tier validation, solver execution, and post-generation actions.
  3. **My Timetables**: Local JSON-based persistent timetable history system (save, open, rename, duplicate, delete).
  4. **View & Edit Timetable**: Authoritative schedule viewer (Class, Teacher, Room grids) and interactive manual editor with real-time conflict detection.
  5. **Experiments & Benchmarks**: Isolated experimental suite evaluating all algorithms and recording genuine metrics without altering user timetables.
  6. **Export**: Synchronized export of the active schedule to CSV, Excel (`.xlsx`), and JSON.
- **Independent Solution Verification**: Every generated schedule is verified against all constraints using `validate_timetable` before being accepted as valid.
- **361 Automated Unit and Integration Tests**: 100% clean test suite with zero failures and zero warnings.

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

## 4. Soft Constraints & Multi-Factor Cost Function

Soft constraints represent educational quality preferences. They do not render a timetable invalid, but lower penalty values indicate higher scheduling quality.

### 4.1 Implemented Soft Constraints (S1–S5)

1. **S1: Minimize Gaps in Class Day ($w_{s1} = 1.0$)**
   - Identifies idle waiting periods between the first and last occupied lecture for each class on each day.
   - *Exclusions*: The class's designated `lunch_slot` and any global `blocked_periods` are not counted as gaps.
   - Penalty: $\sum \text{idle gap periods} \times w_{s1}$.

2. **S2: Spread Subject Lectures Across Days ($w_{s2} = 1.0$)**
   - Encourages spaced learning by penalizing multiple theory lectures of the same subject on the same day.
   - If class $C$ has $k > 1$ theory lectures of subject $S$ on day $D$, penalty is $(k - 1) \times w_{s2}$.
   - *Exemption*: Multi-period lab sessions are atomic and are strictly exempt.

3. **S3: Consecutive Lab Periods & Inter-Session Continuity ($w_{s3} = 1.0$)**
   - By construction, atomic lab sessions occupy consecutive slots $\{p, p + 1\}$.
   - S3 evaluates same-day lab coherence: flags separated placements if multiple lab sessions of the same subject occur on the same day.
   - Does not duplicate hard penalties for lunch (H9) or blocked slots (H6).
   - Penalty: $\sum \text{inter-lab gap periods} \times w_{s3}$.

4. **S4: Teacher Preferred Time Slots ($w_{s4} = 1.0$)**
   - Respects faculty preferences specified in `Teacher.preferred_slots`.
   - Each occupied period of a session placed at $(day, period)$ where $(day, period) \notin \text{teacher.preferred\_slots}$ incurs $1 \times w_{s4}$ penalty.
   - *Exemption*: If a teacher does not specify any preferred slots (`preferred_slots == []`), zero penalty is incurred.

5. **S5: Teacher Maximum Consecutive Lectures ($w_{s5} = 2.0$)**
   - Prevents faculty vocal and physical fatigue by penalizing runs of more than 2 consecutive teaching periods on the same day.
   - For any unbroken sequence of length $L > 2$ consecutive periods, penalty is $(L - 2) \times w_{s5}$.
   - A single 2-period lab incurs 0 penalty ($2 \le 2$). Lunch and empty periods break streaks.

### 4.2 Objective Cost Function

$$\text{Total Cost} = 1000 \times \text{hard violations} + \sum_{i=1}^{5} w_{si} \cdot S_i$$

- Default weights: $w_{s1} = 1.0, w_{s2} = 1.0, w_{s3} = 1.0, w_{s4} = 1.0, w_{s5} = 2.0$.
- Hard violations dominate the cost function ($1000\times$), ensuring that any feasible timetable is strictly preferred over an infeasible one.

---

## 5. Unified Navigation & Workflows

The Streamlit web application (`app.py`) provides an interactive interface organized into six dedicated modules accessible via the sidebar navigation:

### 5.1 Dashboard (`📊 Dashboard`)
- **Purpose**: Central operational hub and overview of system status.
- **Main Workflow**:
  - **Active Timetable Status**: Displays the currently loaded schedule's name, validity status (conflict-free vs. violation count), total cost, and placed session fraction ($N / \text{total}$). Provides one-click action buttons to open the schedule in *View & Edit Timetable* or jump directly to *Export*. Flags unsaved manual modifications.
  - **System Workflow Guide**: Step-by-step guidance on institution configuration, generation, grid inspection, and persistence.
  - **Recent Saved Schedules**: Displays recently saved schedules from local JSON storage with a shortcut to open the full management view in *My Timetables*.

### 5.2 Create Timetable (`🛠️ Create Timetable`)
- **Purpose**: Institutional data configuration, multi-tier validation, and automated schedule generation.
- **Main Workflow**:
  - **Dual Input Modes**:
    - **Quick Setup Wizard (5 Steps)**: Step 1 (*Time Grid & Breaks*, with Express complete `.xlsx` workbook upload and downloadable template), Step 2 (*Classes & Subjects*, theory vs. multi-period lab duration configuration), Step 3 (*Teachers & Matrix*, faculty loads, unavailable periods, preferred time slots, and interactive visual availability grid), Step 4 (*Rooms & Capacities*), and Step 5 (*Solve & Verify*).
    - **Classic Table Editor**: Direct tabular data editing across Classes, Teachers, Subjects, and Rooms dataframes.
  - **Pre-Solve Feasibility Validation**: Validates schema integrity, room capacity limits, total weekly load vs. available slot budget, and teacher availability before running solvers.
  - **Algorithm Execution**: Allows selecting among Plain Backtracking, MRV, MRV+Degree, MRV+Degree+FC, Min-Conflicts, or Hybrid CSP + Simulated Annealing with custom execution parameters (node budgets, iteration counts, time limits, random seeds).
  - **Post-Solve Verification**: Every generated schedule is validated with `validate_timetable`. Upon success, users can save to storage, inspect in grid view, or export.

### 5.3 My Timetables (`📁 My Timetables`)
- **Purpose**: Persistent local timetable lifecycle and history management.
- **Main Workflow**:
  - Reads and lists all schedule records persisted by `_timetable_store` in `saved_timetables/` as JSON files.
  - Displays record metadata: schedule name, solver algorithm used, version tag, last modified timestamp, hard violation count, and total cost.
  - **Actions per Record**:
    - **Set Active**: Loads the schedule into active session state for viewing, editing, or exporting.
    - **Rename**: Renames the record in storage and updates active session state.
    - **Duplicate**: Clones the record with an incremented version counter for branch experimentation.
    - **Delete**: Permanently removes the timetable file from storage with a safety confirmation checkbox.
    - **JSON Backup**: Downloads the complete timetable record JSON.

### 5.4 View & Edit Timetable (`👁️ View & Edit Timetable`)
- **Purpose**: Comprehensive schedule visualization across institutional dimensions and interactive manual adjustment.
- **Main Workflow**:
  - **Status Banner**: Displays schedule name, conflict-free validity badge, objective total cost, soft penalty breakdown (S1–S5), and placed session count.
  - **Schedule Grids**: View schedule filtered by **Class**, **Teacher**, or **Room** with color-coded subject cards, continuation badges for multi-period labs, lunch slot indicators, and collision flags.
  - **Manual Schedule Editor**:
    - Select any scheduled session and reassign it to a new Day, Start Period, or Room.
    - Recomputed immediately using `validate_timetable` and `calculate_cost`. If collisions are introduced, specific hard violations are displayed in real-time.
    - **Save Changes**: Persists edits to the active timetable record in persistent storage.
    - **Revert Changes**: Restores the last clean saved version, discarding manual edits.

### 5.5 Experiments & Benchmarks (`🔬 Experiments & Benchmarks`)
- **Purpose**: Empirical benchmark analysis across standardized academic datasets.
- **Main Workflow**:
  - Inspects results produced by `experiments/run_experiments.py` stored in `experiments/results/`.
  - **Benchmark Summary Statistics**: Displays aggregated performance tables (`experiment_summary.csv`) showing average runtimes, iterations, node counts, backtracks, and total costs across algorithms and datasets.
  - **Convergence Plots**: Displays `min_conflicts_cost_vs_iterations.png` charting conflict decay and objective cost reduction over iterations.
  - **Individual Trial Runs**: Expandable table inspecting per-trial metrics (`experiment_runs.csv`).
  - **State Isolation**: Benchmark execution and review never alter or overwrite active personal timetable records.

### 5.6 Export (`📥 Export`)
- **Purpose**: Exporting the active schedule and dataset in standard administrative formats.
- **Main Workflow**:
  - Displays a clean, tabular view of placed sessions sorted by Class, Day, and Start Period.
  - Warns if the active schedule contains unresolved manual conflicts.
  - **Export Formats**:
    - **CSV (`timetable.csv`)**: Flat spreadsheet for integration with school management systems.
    - **Excel (`timetable.xlsx`)**: Formatted workbook generated via `openpyxl`.
    - **Dataset JSON (`dataset.json`)**: Raw dataset schema file for portability or archival.

---

## 6. Solver Algorithms

### 6.1 Systematic Backtracking Solvers (`core/backtracking.py`)

1. **Plain Backtracking (Baseline)**: Systematic Depth-First Search in static variable generation order with dynamic collision checks.
2. **Backtracking + MRV (Minimum Remaining Values)**: Dynamically selects the variable with the fewest remaining legal placements.
3. **Backtracking + MRV + Degree Heuristic**: Selects variables with minimum legal domain size, breaking ties using dynamic constraint degree.
4. **Backtracking + MRV + Degree + Forward Checking (`mrv_degree_fc`)**: Maintains active domains in `current_domains`, prunes conflicting values 1 step ahead upon each assignment, and backtracks immediately on domain wipeout.
5. **Quality-Aware Backtracking (`optimize_quality=True`)**: Orders candidate domain values by soft-constraint delta to bias search toward higher-quality placements.

### 6.2 Min-Conflicts Local Search (`core/min_conflicts.py`)

- **Heuristic Repair Paradigm**: Starts from a complete initial placement drawn from valid unary domains and iteratively repairs the variable with the highest conflict count.
- **Atomic Conflict Scoring**: Evaluates exact binary clashes across H1 (Teacher), H2 (Class), and H3 (Room) with self-exclusion.
- **Plateau Escaping**: Random walk probability $p_{\text{walk}} = 0.05$, stagnation detection with perturbation after 150 non-improving steps, and non-stationary tie-breaking.
- **Optional Two-Stage Soft Refinement (`refine_soft_costs=True`)**: Once 0 hard violations are reached, local moves strictly preserving 0 hard conflicts are executed to minimize soft penalties.

### 6.3 Hybrid CSP + Simulated Annealing (`core/simulated_annealing.py`)

- **Two-Phase Architecture**:
  1. **Phase 1 (Feasibility)**: Runs Backtracking with MRV + Degree + FC to find a guaranteed feasible solution ($H = 0$).
  2. **Phase 2 (Optimization)**: Explores neighboring feasible timetables using single-variable moves and compatible 2-variable swaps.
- **Metropolis Acceptance Criterion**:
  - Downhill moves ($\Delta \text{Cost} < 0$) are always accepted.
  - Uphill moves ($\Delta \text{Cost} > 0$) are accepted with probability $P = \exp(-\Delta \text{Cost} / T)$.
- **Cooling Schedule**: Geometric cooling $T_{k+1} = \alpha \cdot T_k$ with restart on stagnation.
- **Hard-Constraint Guarantee**: Neighborhood moves that introduce any hard violation are rejected outright.

---

## 7. Benchmark Datasets

| Dataset | Classes | Teachers | Rooms | Grid | Blocked Slots | Total Sessions | Characteristics |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Small Dataset** | 3 | 6 | 4 (3 CR, 1 Lab) | 5 days $\times$ 6 periods | None | 40 | Low-contention baseline; fast feasibility proof. |
| **Medium Tight Dataset** | 4 | 4 | 3 (2 CR, 1 Lab) | 5 days $\times$ 6 periods | Assembly (Mon P1, Fri P6) | 40 | High contention: 4 classes share only 2 classrooms, 1 shared lab, tight teacher availability. |
| **Large Dataset** | 8 | 15 | 8 (6 CR, 2 Labs) | 5 days $\times$ 7 periods | None | 105 | Academic institution scale; tests scalability and lookahead pruning. |

---

## 8. Actual Benchmark Measurements

Measured on Python 3.14 across 10 trials per stochastic algorithm using master random seed 42:

### 8.1 Systematic Backtracking Benchmark

| Dataset | Algorithm | Status | Runtime | Nodes Explored | Assignments Tried | Backtracks | Hard Violations | Soft Breakdown (S1 / S2 / S3 / S4 / S5) | Total Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | Plain Backtracking | `SOLVED` | **0.0008s** | 40 | 642 | 0 | 0 | 1.0 / 25.0 / 0.0 / 0.0 / 8.0 | 34.0 |
| **Small** | Backtracking + MRV | `SOLVED` | **0.0169s** | 40 | 753 | 0 | 0 | 0.0 / 21.0 / 0.0 / 0.0 / 5.0 | 26.0 |
| **Small** | Backtracking + MRV + Degree | `SOLVED` | **0.0175s** | 40 | 753 | 0 | 0 | 0.0 / 21.0 / 0.0 / 0.0 / 5.0 | 26.0 |
| **Small** | Backtracking + MRV + Degree + FC | `SOLVED` | **0.0042s** | 40 | **39** | 0 | 0 | 0.0 / 21.0 / 0.0 / 0.0 / 5.0 | 26.0 |
| **Medium Tight** | Plain Backtracking | `SOLVED` | **0.0007s** | 41 | 477 | 0 | 0 | 5.0 / 14.0 / 0.0 / 35.0 / 4.0 | 58.0 |
| **Medium Tight** | Backtracking + MRV | `SOLVED` | **0.0121s** | 41 | 479 | 0 | 0 | 3.0 / 11.0 / 0.0 / 34.0 / 6.0 | 54.0 |
| **Medium Tight** | Backtracking + MRV + Degree | `SOLVED` | **0.0111s** | 41 | 479 | 0 | 0 | 3.0 / 11.0 / 0.0 / 34.0 / 6.0 | 54.0 |
| **Medium Tight** | Backtracking + MRV + Degree + FC | `SOLVED` | **0.0035s** | 41 | **40** | 0 | 0 | 3.0 / 11.0 / 0.0 / 34.0 / 6.0 | 54.0 |
| **Large** | Plain Backtracking | `SOLVED` | **0.0043s** | 105 | 4,029 | 0 | 0 | 4.0 / 46.0 / 0.0 / 0.0 / 17.0 | 67.0 |
| **Large** | Backtracking + MRV | `SOLVED` | **0.2805s** | 105 | 3,874 | 0 | 0 | 4.0 / 48.0 / 0.0 / 0.0 / 23.0 | 75.0 |
| **Large** | Backtracking + MRV + Degree | `SOLVED` | **0.2841s** | 105 | 3,874 | 0 | 0 | 4.0 / 48.0 / 0.0 / 0.0 / 23.0 | 75.0 |
| **Large** | Backtracking + MRV + Degree + FC | `SOLVED` | **0.0582s** | 105 | **104** | 0 | 0 | 4.0 / 48.0 / 0.0 / 0.0 / 23.0 | 75.0 |

### 8.2 Heuristic Local Search & Optimization Benchmark (10 Randomized Trials)

| Dataset | Algorithm | Success Rate | Avg Runtime | Avg Iterations | Avg Hard Violations | Avg Soft Breakdown (S1 / S2 / S3 / S4 / S5) | Avg Total Cost |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** | Min-Conflicts | **100%** | **0.0060s** | 15.7 | 0.0 | 9.5 / 8.1 / 0.0 / 0.0 / 1.2 | 18.8 |
| **Small** | **Hybrid CSP + SA** | **100%** | **0.1743s** | 2000.0 | 0.0 | **4.1 / 5.4 / 0.0 / 0.0 / 0.5** | **10.0** (-61.5%) |
| **Medium Tight** | Min-Conflicts | **100%** | **0.0078s** | 19.6 | 0.0 | 10.1 / 5.1 / 0.0 / 35.8 / 2.0 | 53.0 |
| **Medium Tight** | **Hybrid CSP + SA** | **100%** | **0.1266s** | 2000.0 | 0.0 | **4.4 / 3.8 / 0.0 / 35.2 / 1.4** | **44.8** (-17.0%) |
| **Large** | Min-Conflicts | **100%** | **0.0384s** | 42.2 | 0.0 | 40.9 / 14.4 / 0.0 / 0.0 / 2.6 | 57.9 |
| **Large** | **Hybrid CSP + SA** | **100%** | **0.4557s** | 2000.0 | 0.0 | **26.7 / 13.8 / 0.0 / 0.0 / 3.4** | **43.9** (-41.5%) |

### 8.3 Algorithmic Insights: Feasibility vs. Optimization

1. **Pruning Power of Forward Checking**:
   - Because the timetable search tree is wide with many feasible branches, systematic Backtracking solvers find an initial path with 0 backtracks.
   - However, **Plain Backtracking evaluates hundreds to thousands of candidate assignments** that are doomed to fail (642 on Small, 4,029 on Large).
   - **Forward Checking prunes invalid placements 1 step ahead**, slashing candidate assignments by **94% on Small** (642 $\to$ 39) and **97.4% on Large** (4,029 $\to$ 104).
2. **Feasibility vs. Optimization Tradeoff**:
   - Backtracking and basic Min-Conflicts are **pure satisfaction algorithms**: their objective stops the moment hard violations reach zero. Consequently, their resulting timetables have arbitrary soft costs (e.g., 26 on Small, 75 on Large).
   - **Hybrid CSP + Simulated Annealing bridges feasibility and optimization**: it seeds the annealing process with an independently validated $H = 0$ CSP schedule and systematically anneals soft penalties down to **10.0 on Small** and **43.9 on Large** without violating any hard constraints.

---

## 9. Installation & Usage

### 9.1 Setup
Requires Python 3.10+:

```bash
cd D:\Timetable

# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Launching the Web Application
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

### 9.3 Running Automated Experiments
```bash
python experiments/run_experiments.py
```
Outputs:
- `experiments/results/experiment_runs.csv`: Per-trial metrics across all configurations and random seeds.
- `experiments/results/experiment_summary.csv`: Aggregated means, success rates, and runtimes.
- `experiments/results/min_conflicts_cost_vs_iterations.png`: Convergence plot showing conflict decay and cost profiles.

### 9.4 Running the Test Suite
```bash
python -m pytest -q
```
Current execution result:
```
........................................................................ [ 20%]
........................................................................ [ 40%]
........................................................................ [ 60%]
........................................................................ [ 80%]
........................................................................ [100%]
361 passed in 6.58s

```

---

## 10. Project Structure

```
Timetable/
├── core/
│   ├── __init__.py             # Package exports (Solvers, Dataclasses, Validators)
│   ├── models.py               # Dataclasses, JSON loading, schema & feasibility checks
│   ├── sessions.py             # SessionVar, Placement, Timetable container
│   ├── constraints.py          # H1-H9 + STRUCT validators, validate_timetable
│   ├── cost.py                 # S1-S5 soft penalties, CostBreakdown, calculate_cost
│   ├── backtracking.py         # Plain, MRV, MRV+Degree, MRV+Degree+FC solvers
│   ├── min_conflicts.py        # Min-Conflicts Local Search & two-stage refinement
│   ├── simulated_annealing.py  # Hybrid CSP + Simulated Annealing optimizer
│   ├── custom_builder.py       # Custom timetable parsing, multi-tier validation
│   ├── bulk_importer.py        # TSV, CSV, and XLSX bulk data ingestion
│   └── storage.py              # Persistent atomic JSON storage for My Timetables
├── datasets/
│   ├── small_dataset.json      # 3 classes, 6 teachers, 4 rooms, 5x6 grid (40 sessions)
│   ├── medium_tight_dataset.json# 4 classes, 4 teachers, 3 rooms, 5x6 grid with blocked slots
│   ├── large_dataset.json      # 8 classes, 15 teachers, 8 rooms, 5x7 grid (105 sessions)
│   ├── small_feasible_schedule.json # Proof schedule
│   └── large_feasible_schedule.json # Proof schedule
├── experiments/
│   ├── run_experiments.py      # Benchmark runner with normalized metrics & master seeds
│   └── results/                # CSVs and PNG convergence plot
├── saved_timetables/           # Persistent local JSON storage for user schedules
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_models.py          # Model validation tests
│   ├── test_sessions.py        # Session generation and placement tests
│   ├── test_constraints.py     # Isolated H1-H9 violation tests
│   ├── test_cost.py            # S1-S5 penalty tests
│   ├── test_datasets.py        # Dataset structure and feasibility tests
│   ├── test_feasibility.py     # Proof schedule verification tests
│   ├── test_backtracking.py    # Backtracking solver tests (55 tests)
│   ├── test_min_conflicts.py   # Min-Conflicts solver tests (19 tests)
│   ├── test_simulated_annealing.py # Hybrid SA solver tests (6 tests)
│   ├── test_variable_lab_durations.py # Variable-duration lab tests (11 tests)
│   ├── test_experiments.py     # Experiment runner & normalized metrics tests (37 tests)
│   ├── test_grid_ui.py         # Grid UI rendering & special char tests (58 tests)
│   ├── test_app.py             # Streamlit app tests (55 tests)
│   ├── test_custom_builder.py  # Custom builder tests (33 tests)
│   ├── test_storage.py         # Timetable store tests (7 tests)
│   └── test_refactored_workflow.py # Workflow and regression tests (11 tests)

├── app.py                      # Streamlit application with UI grid rendering
├── run_solver.py               # CLI entry point
├── requirements.txt            # Python dependencies
└── README.md
```
