# AI-Based Timetable Scheduler (Phase 1: CSP Formulation)

**Course**: Artificial Intelligence (B.Tech, Year 2)  
**Phase**: Phase 1 — Data Models, Session Generation, Constraint Validators, Cost Function, Feasible Datasets & Test Suite  
**Stack**: Python 3.10+, `pytest`, JSON datasets  

---

## 1. Project Purpose & Scope

The purpose of this project is to model and solve college timetable scheduling as a **Constraint Satisfaction Problem (CSP)**. In an academic institution, lectures and laboratory sessions must be assigned to days, time slots, and rooms such that neither teachers, student batches, nor physical venues clash.

### Phase 1 Scope (Delivered)
- **Data Models & Schema Validation**: Strongly-typed dataclasses for TimeStructure, Subjects, Teachers, Classes, Rooms, and full Datasets.
- **CSP Problem Formulation**: Atomic CSP variables (`SessionVar`) representing teaching sessions, domain values (`Placement`), and timetable assignment mappings.
- **Hard Constraint Validators (H1–H9 + STRUCT)**: Independent validation functions with explicit, documented counting rules.
- **Soft Constraint Evaluators (S1–S3)**: Quantitative penalty evaluators with configurable weights and a composite objective cost function.
- **Realistic Benchmarks**: `small_dataset.json` (3 classes, 6 teachers, 4 rooms, 5×6 grid) and `large_dataset.json` (8 classes, 15 teachers, 8 rooms, 5×7 grid).
- **Constructive Feasibility Evidence**: Hand-constructed, conflict-free schedules (`small_feasible_schedule.json` and `large_feasible_schedule.json`) proving mathematical solvability without any solver.
- **Comprehensive Unit Tests**: Full test suite across all models, constraints, edge cases, and datasets.

### Strict Non-Goals (Deferred to Phase 2)
As mandated by the Phase 1 specification:
- **No search algorithms**: Backtracking, MRV, Forward Checking, Min-Conflicts, Simulated Annealing, and Hill Climbing are strictly deferred to Phase 2.
- **No external solvers**: `python-constraint` is not used.
- **No User Interface**: Streamlit / Tkinter frontend is deferred to Phase 2.
- **No Soft Constraints S4/S5**: S4 (Teacher preferences) and S5 (Teacher consecutive lecture limits) are deferred.
- **No fake experimental data**: No benchmark performance graphs or placeholder timing tables.

### Academic Integrity Disclosure
In compliance with assignment regulations, AI assistance was utilized as a pair-programming assistant to help architect dataclasses, implement validation logic, and design deterministic placement patterns. Every module, constraint formulation, and line of code is structured for complete student comprehension and oral defense in a viva examination.

---

## 2. CSP Formulation

### 2.1 Variables ($X$) — `SessionVar`
Each variable represents an atomic lecture or laboratory session that must be placed on the schedule grid:
$$\text{SessionVar} = (\text{session ID}, \text{class ID}, \text{subject ID}, \text{session no}, \text{teacher ID}, \text{duration}, \text{session type})$$

- **Theory Session**: Duration = 1 period. A subject with 3 weekly hours generates 3 distinct `SessionVar` instances (`session_no` $\in \{1, 2, 3\}$).
- **Lab Session**: Duration = `lab_duration` (e.g. 2 periods). A lab with 2 weekly hours generates **one single session** occupying 2 consecutive periods. It is NOT modeled as two disconnected 1-hour sessions.

### 2.2 Domain Values ($D$) — `Placement`
A placement value represents a specific coordinate in time and space:
$$\text{Placement} = (\text{day}, \text{start period}, \text{room ID})$$
- $\text{day} \in \{0, \dots, \text{days per week} - 1\}$
- $\text{start period} \in \{0, \dots, \text{periods per day} - 1\}$
- A session occupies periods $\{p, p + 1, \dots, p + \text{duration} - 1\}$ on that day.
- $\text{room ID} \in \text{rooms}$

### 2.3 Assignment & Timetable
A timetable assignment is a mapping $\mathcal{A}: \text{SessionVar} \to \text{Placement} \cup \{\text{None}\}$.
- `is_complete()`: Returns `True` iff every required variable has a non-`None` placement.
- `is_valid()`: Returns `True` iff `is_complete() == True` AND `hard_violation_count == 0`. An empty or partial timetable is **never** valid.

---

## 3. Hard Constraints (H1–H9 + STRUCT)

Hard constraints represent physical and institutional impossibilities. A timetable with any hard violation is invalid.

### Exact Counting Rules

| Constraint | Name | Exact Counting Rule | Viva Explanation |
|---|---|---|---|
| **H1** | Teacher Clash | **1 violation per conflicting pair of sessions per overlapping period**. If Teacher T is scheduled in 2 sessions simultaneously for 1 period = 1 violation. If a 2-period lab overlaps with another session on both periods = 2 violations. | Binary constraint. A teacher cannot physically be in two classrooms at the same time. |
| **H2** | Class Clash | **1 violation per conflicting pair of sessions for the same class per overlapping period**. | Binary constraint. A student batch cannot attend two lectures at the same time. |
| **H3** | Room Clash | **1 violation per conflicting pair of sessions in the same room per overlapping period**. | Binary constraint. A room cannot host two distinct sessions simultaneously. |
| **H4** | Room Capacity | **1 violation per placed session** where `room.capacity < class.strength`. | Unary constraint. Overcrowded classrooms cannot seat all enrolled students. |
| **H5** | Room Type | **1 violation per placed session** in an incompatible room (`lab` in classroom, or `theory` in lab). | Unary constraint. Labs require specialized equipment; theory lectures require standard lecture halls. |
| **H6** | Teacher Availability & Blocked Slots | **1 violation per occupied slot** falling on a teacher's `unavailable_slots` OR a global `blocked_periods` (e.g., college assembly). | Unary constraint on time coordinates. |
| **H7** | Weekly Syllabus Hours | **$|\text{assigned periods} - \text{required periods}|$ per (class, subject)**. Missing and extra periods both count. Unassigned sessions contribute directly as missing periods. | Quota constraint ensuring complete syllabus coverage. An empty timetable has violations equal to total required periods. |
| **H8** | Teacher Max Weekly Load | **$\max(0, \text{assigned periods} - \text{max weekly load})$** per teacher. | Workload limit protecting faculty from overwork (a 2-period lab adds 2 periods towards load). |
| **H9** | Class Lunch Break | **1 violation per occupied slot** falling on `class.lunch_slot`. | Institutional policy ensuring students have a daily dedicated lunch break. |
| **STRUCT** | Structural Validity | **1 violation per occurrence**: out-of-bounds day/period, multi-period session overflowing the end of the day, unknown entity ID, or teacher assigned to a subject they are not qualified to teach. | Ensures data integrity and prevents silent out-of-bounds corruption. |

---

## 4. Soft Constraints & Cost Function

Soft constraints represent quality preferences. They do not render a timetable invalid, but higher penalties indicate poorer scheduling quality.

### 4.1 Implemented Soft Constraints (S1–S3)
Soft penalties are computed strictly on placed sessions:
- **S1: Minimize Gaps in Class Day**
  - Identifies idle waiting periods between the first and last occupied lecture for each class on each day.
  - **Exclusions**: The class's designated `lunch_slot` and any global `blocked_periods` are **not** counted as gaps.
  - Penalty: $\text{idle gap periods} \times w_{s1}$.
- **S2: Spread Subject Lectures Across Days**
  - Encourages spaced learning by penalizing multiple theory lectures of the same subject on the same day.
  - If class $C$ has $k > 1$ theory lectures of subject $S$ on day $D$, penalty is $(k - 1) \times w_{s2}$.
  - **Exemption**: Lab sessions are strictly exempt (a 2-period lab is 1 session and is not a duplicate).
- **S3: Consecutive Lab Periods**
  - By construction, an atomic lab session occupies consecutive slots $\{p, p + 1\}$.
  - S3 explicitly flags and penalizes:
    1. A lab placement that is interrupted by a class lunch break or a global blocked period.
    2. Multiple lab sessions of the same subject on the same day placed with idle periods between them.
  - Penalty: $\text{interrupted or separated slots} \times w_{s3}$.

### 4.2 Objective Cost Function
$$\text{Total Cost} = 1000 \times \text{hard violations} + (w_{s1} \cdot S_1 + w_{s2} \cdot S_2 + w_{s3} \cdot S_3)$$
- Default weights: $w_{s1} = 1, w_{s2} = 1, w_{s3} = 1$ (customizable via `SoftWeights`).
- Hard violations dominate the cost function ($1000 \times$), ensuring that any feasible timetable is strictly preferred over an infeasible one.
- Returns a complete `CostBreakdown` exposing hard cost, individual soft costs, and validity flags.

---

## 5. Dataset Specifications & Schemas

### 5.1 Dataset Summary

| Metric | Small Dataset (`small_dataset.json`) | Large Dataset (`large_dataset.json`) |
|---|---|---|
| **Classes** | 3 (`CSE_A`, `CSE_B`, `ECE_A`) | 8 (`CSE_2A`, `CSE_2B`, `AI_2A`, `AI_2B`, `ECE_2A`, `ECE_2B`, `IT_2A`, `IT_2B`) |
| **Teachers** | 6 (`T1` to `T6`) | 15 (`T01` to `T15`) |
| **Rooms** | 4 (3 Classrooms + 1 Lab) | 8 (6 Classrooms + 2 Labs) |
| **Grid Dimensions** | 5 days $\times$ 6 periods (30 slots/room) | 5 days $\times$ 7 periods (35 slots/room) |
| **Lunch Break** | Staggered (Period 3 and Period 4) | Staggered (Period 3 and Period 4) |
| **Global Breaks** | Monday Period 0 (Assembly) | Monday Period 0 (Assembly) |
| **Total Sessions** | 39 sessions (42 periods) | 104 sessions (112 periods) |

### 5.2 Programmatic Feasibility Sanity Checks
Both datasets pass programmatic mathematical solvability assertions in `core/models.py` (`check_dataset_feasibility`):
1. **Teacher Qualification**: Every subject has $\ge 1$ qualified teacher; every class requirement binds a qualified teacher.
2. **Workload Feasibility**: For every teacher, $\text{assigned periods} \le \text{max weekly load}$ and $\le \text{available slots}$.
3. **Class Feasibility**: For every class, $\text{required periods} \le (\text{days} \times \text{periods} - \text{lunch} - \text{blocked})$.
4. **Room Capacity**: For every class, there exists at least one classroom and at least one lab with $capacity \ge strength$.
5. **Aggregate Venue Capacity**: Total required theory and lab periods are strictly less than total classroom and lab slot capacities.

### 5.3 JSON Schema Sample (Small Snippet)
```json
{
  "name": "Small Academic Dataset",
  "time_structure": {
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "periods_per_day": 6,
    "blocked_periods": [[0, 0]]
  },
  "subjects": [
    {"id": "CS101", "name": "Programming Fundamentals", "weekly_periods": 4, "type": "theory"},
    {"id": "CS101_LAB", "name": "Programming Lab", "weekly_periods": 2, "type": "lab", "lab_duration": 2}
  ],
  "teachers": [
    {"id": "T1", "name": "Prof. Alan Turing", "subject_ids": ["CS101"], "unavailable_slots": [[4, 5]], "max_weekly_load": 14}
  ],
  "rooms": [
    {"id": "CR1", "name": "Classroom 1", "capacity": 40, "type": "classroom"},
    {"id": "LAB1", "name": "Computing Lab 1", "capacity": 35, "type": "lab"}
  ],
  "classes": [
    {
      "id": "CSE_A",
      "name": "Computer Science A",
      "strength": 30,
      "lunch_slot": 3,
      "subjects": [
        {"subject_id": "CS101", "teacher_id": "T1", "weekly_periods": 4},
        {"subject_id": "CS101_LAB", "teacher_id": "T6", "weekly_periods": 2}
      ]
    }
  ]
}
```

---

## 6. Feasibility Evidence (Constructive Proofs)

To satisfy the academic requirement that both datasets are provably solvable **without running a solver**:

### Construction Strategy
1. **Time-Band Decomposition**:
   - Classes are partitioned into two groups based on lunch times.
   - Group 1 (CSE/AI) operates in morning theory periods $\{1, 2, 4\}$ with lunch at 3.
   - Group 2 (ECE/IT) operates in afternoon theory periods $\{3, 5, 6\}$ with lunch at 4.
   - This orthogonal temporal separation eliminates any cross-group teacher or room clashes.
2. **Paired Latin Shift**:
   - Within each group, classes sharing teachers are scheduled using cyclic permutations across days.
   - When Class 1 has Subject $A$ at slot $(d, p)$, Class 2 has Subject $B$.
3. **Atomic Lab Windows**:
   - Labs are allocated to dedicated computing and electronics lab rooms during designated windows without intersecting class lunch breaks.

### Automated Verification
The hand-constructed schedules are saved in:
- `datasets/small_feasible_schedule.json`
- `datasets/large_feasible_schedule.json`

Automated unit tests in `tests/test_feasibility.py` load both schedules and assert:
```python
assert result.is_complete is True
assert result.is_valid is True
assert result.hard_violation_count == 0
```
Both schedules achieve **0 hard violations**, proving feasibility beyond doubt.

---

## 7. Backtracking CSP Solvers (`core/backtracking.py`)

The project implements four systematic constructive search configurations for the CSP timetable problem:
1. **Plain Backtracking (Baseline)**: Evaluates variables in static generation order with dynamic collision checks.
2. **Backtracking + MRV (Minimum Remaining Values)**: Dynamically selects the variable with the fewest remaining legal placements.
3. **Backtracking + MRV + Degree Heuristic**: Dynamically selects variables with minimum remaining legal placements, breaking ties using the variable with the highest degree (most constraint relationships with other unassigned variables).
4. **Backtracking + MRV + Degree + Forward Checking (`mrv_degree_fc`)**: Combines MRV variable selection and Degree tie-breaking with active Forward Checking (FC). Dynamically maintains legal domains of unassigned variables and prunes conflicting placements 1 step ahead after each assignment.

### 7.1 How the Heuristics & Forward Checking Work

1. **Variable Generation**: `generate_session_variables(dataset)` generates deterministic session variables in static order.
2. **Unary Domain Pruning**: `get_valid_domain_for_session(session, dataset)` computes all candidate placements satisfying unary constraints:
   - Room type matches (`theory` $\to$ `classroom`, `lab` $\to$ `lab`).
   - Room capacity $\ge$ class strength.
   - Teacher availability (no placement on teacher's `unavailable_slots`).
   - Class lunch breaks (no placement during class `lunch_slot`).
   - Global blocked periods (no placement during college breaks).
   - Daily boundary fitting (`start_period + duration <= periods_per_day`).
3. **Empty Domain Check**: If any variable has an empty initial domain, the problem is immediately proven **unsatisfiable** without wasted search.
4. **Dynamic Variable Selection**:
   - **Plain Backtracking (`use_mrv=False, use_degree=False, use_fc=False`)**: Evaluates unassigned variables in static dataset generation order.
   - **Backtracking + MRV (`use_mrv=True, use_degree=False, use_fc=False`)**:
     - At each recursive decision step, scans all currently unassigned variables.
     - Counts the number of currently legal placements for each variable under dynamic collision constraints (H1–H3).
     - Selects the variable with the **minimum remaining legal values** (MRV).
     - Ties broken by static generation order.
   - **Backtracking + MRV + Degree (`use_mrv=True, use_degree=True, use_fc=False`)**:
     1. **Primary**: Selects variable(s) with the minimum currently legal domain size (MRV).
     2. **Secondary (Degree Heuristic)**: If multiple variables tie on domain size, selects the variable participating in the **highest number of binary constraint relationships with other unassigned variables**.
     3. **Tertiary (Deterministic Tie-Breaker)**: If still tied, selects the variable with the smallest original generation index.
   - **Backtracking + MRV + Degree + Forward Checking (`use_mrv=True, use_degree=True, use_fc=True`)**:
     1. **Primary (O(1) MRV)**: Selects the unassigned variable with the smallest active domain size directly from `current_domains`.
     2. **Secondary (Degree Heuristic)**: Breaks ties using dynamic degree among unassigned variables.
     3. **Tertiary (Deterministic Tie-Breaker)**: Breaks remaining ties using original generation index.
5. **How Forward Checking Works**:
   - During search, `current_domains` maintains the active, pruned set of legal placements for each unassigned variable.
   - When a placement $P$ is assigned to variable $V$:
     1. **Identify Affected Variables**: Finds all unassigned variables $U$ that share a constraint edge with $V$ in the precomputed binary constraint graph.
     2. **Prune Incompatible Placements**: Removes any placement $P_U \in Domain(U)$ that conflicts with $(V, P)$ under H1 (Teacher), H2 (Class), or H3 (Room).
     3. **Multi-Period Lab Accounting**: Properly checks 1D interval overlap across all occupied periods (e.g., a 2-period lab at periods 1–2 prunes placements at both period 1 and period 2).
     4. **Domain Wipeout Detection**: If any unassigned variable's domain becomes empty ($|Domain(U)| = 0$), Forward Checking detects this failure immediately, aborts the current branch, restores all modified domains, and backtracks without expanding any child nodes.
     5. **State Restoration**: When search backtracks out of an assignment, all pruned placements are restored cleanly to their parent state domains.
   - *Theoretical Limitation*: Forward Checking checks the direct consequences of the *current* assignment against unassigned variables (1-step lookahead). It does **not** guarantee global consistency, because it does not propagate constraints between two unassigned variables (which would require AC-3 or MAC).
6. **Exact Definition of Degree**:
   Two distinct session variables $V_A$ and $V_B$ share a binary constraint edge in the CSP graph if there exists at least one pair of placements $(p_A \in Domain(V_A), p_B \in Domain(V_B))$ that can overlap in time and conflict on:
   - **H1 (Teacher)**: `V_A.teacher_id == V_B.teacher_id`.
   - **H2 (Class)**: `V_A.class_id == V_B.class_id`.
   - **H3 (Room)**: `p_A.room_id == p_B.room_id`.
   Multi-period lab durations are accounted for across all occupied periods. Variables with disjoint rooms (theory vs. lab) and distinct teachers/classes have **no edge** between them.
7. **Final Authority**: When all variables are assigned, `validate_timetable(tt)` from `core.constraints` runs full verification against all constraints H1–H9 and STRUCT.

### 7.2 Session Variable vs. Occupied Period
| Concept | Definition | Small Dataset | Large Dataset |
|---|---|---|---|
| **Session Variable (`SessionVar`)** | Atomic CSP decision variable. A 1-hour lecture is 1 variable. A 2-hour lab is **1 variable** (not 2). | **39 variables** | **104 variables** |
| **Occupied Period** | Physical period units consumed on the schedule grid ($Session \times Duration$). | **42 periods** | **112 periods** |

### 7.3 Benchmark Comparison (Real Execution Results)

The table below records real execution metrics comparing all four configurations across identical datasets on Python 3.14:

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

> **Honest Performance & Overhead Analysis**:
> 1. **Massive Reduction in Assignments Tried**:  
>    In Plain and MRV search, hundreds or thousands of candidate placements are tested against dynamic collision sets during search (642/753 on small; 4,029/3,874 on large). With Forward Checking, incompatible placements are pruned from unassigned domains immediately upon each assignment. As a result, candidate trial count drops to **39 on small** and **104 on large** — exactly 1 candidate tried per variable!
> 2. **Significant Speedup over MRV / MRV + Degree**:  
>    Forward Checking accelerates MRV search dramatically: from **0.0180s down to 0.0051s** on small (3.5× faster) and from **0.3271s down to 0.0670s** on large (4.8× faster). This speedup occurs because MRV domain length lookups become $O(1)$ operations (`len(current_domains[v])`) rather than re-evaluating dynamic collision checks across all placements of all unassigned variables at every step.
> 3. **Comparison with Plain Backtracking (Bookkeeping Trade-Off)**:  
>    On these two benchmark datasets, Plain Backtracking solves the problem in **0.0008s** (small) and **0.0046s** (large) with **0 backtracks**. Because the problem instances have sufficient room slack and no bottlenecks, Plain Backtracking happens to find a valid leaf on its very first descent. On instances where 0 backtracks occur, any lookahead heuristic (MRV, Degree, Forward Checking) incurs bookkeeping overhead (allocating pruned lists, maintaining sets) without saving node expansions.
> 4. **Where Forward Checking Shines (Constrained Instances)**:  
>    When problems are tightly constrained or contain impossible subtrees (e.g., synthetic pigeonhole benchmarks and teacher conflicts), Plain Backtracking exhausts combinatorial paths exploring dead ends. Forward Checking detects the resulting domain wipeouts 1 step ahead and backtracks immediately, pruning exponential subtrees.

### 7.4 Running the Solvers

#### Option A: Direct CLI Runner (`run_solver.py`)
```bash
# 1. Plain Backtracking on small dataset (default)
python run_solver.py --dataset small --algorithm plain

# 2. Backtracking + MRV on small dataset
python run_solver.py --dataset small --algorithm mrv
python run_solver.py --dataset small --mrv

# 3. Backtracking + MRV + Degree on small dataset
python run_solver.py --dataset small --algorithm mrv_degree
python run_solver.py --dataset small --degree

# 4. Backtracking + MRV + Degree + Forward Checking on small dataset
python run_solver.py --dataset small --algorithm mrv_degree_fc
python run_solver.py --dataset small --fc

# Forward Checking on large dataset
python run_solver.py --dataset large --algorithm mrv_degree_fc

# Save solved schedule to JSON file
python run_solver.py --dataset small --algorithm mrv_degree_fc --output solution.json
```

#### Option B: Python Script
```python
from pathlib import Path
from core.models import load_dataset_from_file
from core.backtracking import (
    solve_backtracking,
    solve_backtracking_mrv,
    solve_backtracking_mrv_degree,
    solve_backtracking_mrv_degree_fc,
)

dataset = load_dataset_from_file(Path("datasets/small_dataset.json"))

# 1. Plain Backtracking
res_plain = solve_backtracking(dataset, use_mrv=False, use_degree=False, use_fc=False)

# 2. Backtracking + MRV
res_mrv = solve_backtracking_mrv(dataset)

# 3. Backtracking + MRV + Degree
res_degree = solve_backtracking_mrv_degree(dataset)

# 4. Backtracking + MRV + Degree + Forward Checking
res_fc = solve_backtracking_mrv_degree_fc(dataset)

print(f"Plain:  cost={res_plain.total_cost}, time={res_plain.runtime_seconds:.4f}s, tried={res_plain.assignments_tried}")
print(f"MRV:    cost={res_mrv.total_cost}, time={res_mrv.runtime_seconds:.4f}s, tried={res_mrv.assignments_tried}")
print(f"Degree: cost={res_degree.total_cost}, time={res_degree.runtime_seconds:.4f}s, tried={res_degree.assignments_tried}")
print(f"FC:     cost={res_fc.total_cost}, time={res_fc.runtime_seconds:.4f}s, tried={res_fc.assignments_tried}")
```

---

## 8. Min-Conflicts Local Search Solver (`core/min_conflicts.py`)

In addition to constructive backtracking search, the project implements **Algorithm 2: Min-Conflicts**, a heuristic local-search algorithm specifically designed for Constraint Satisfaction Problems (Minton et al., 1992; Russell & Norvig AIMA).

### 8.1 How Min-Conflicts Works

1. **Complete Assignment with Unary Compatibility**:
   - Unlike constructive backtracking (which starts with an empty timetable and assigns one variable at a time), Min-Conflicts begins with a **complete assignment** where every session variable is placed.
   - All candidate placements are drawn exclusively from `get_valid_domain_for_session(session, dataset)`, guaranteeing that unary constraints (H4 Room Capacity, H5 Room Type, H6 Teacher Availability & Blocked Slots, H9 Class Lunch Break, and STRUCT Bounds) are **never violated**.
   - Initial assignment strategies:
     * `"random"` (default): Uniformly selects a legal placement for each variable, producing an initial state containing conflicts for local repair.
     * `"greedy"`: Sequentially assigns each variable to the placement that minimizes conflicts with already-assigned variables.

2. **Conflict Scoring Function**:
   - For a candidate placement $P$ of variable $V$, the conflict score calculates the exact number of atomic binary collisions introduced with all **other** placed variables $U \neq V$ across all occupied slots $(day, period)$:
     * **H1 (Teacher Clash)**: Number of other sessions sharing `V.teacher_id` at that slot.
     * **H2 (Class Clash)**: Number of other sessions sharing `V.class_id` at that slot.
     * **H3 (Room Clash)**: Number of other sessions sharing `P.room_id` at that slot.
   - **Multi-Period Lab Accounting**: For multi-period sessions (e.g., 2-period labs), conflicts are evaluated across all occupied slots: `(day, start_period)` and `(day, start_period + 1)`.
   - **Self-Exclusion**: Variable $V$ is temporarily removed from the occupancy counters before evaluating candidate placements, preventing $V$ from conflicting with itself.

3. **Iterative Local Repair**:
   - At each iteration:
     1. Identifies all currently conflicted variables ($\text{conflicts}(V) > 0$).
     2. If no conflicted variables remain, the schedule has 0 hard violations and search terminates successfully.
     3. Selects a conflicted variable (randomly or deterministically).
     4. Evaluates all valid domain placements for that variable.
     5. Reassigns the variable to the placement that **minimizes the conflict score**.

4. **Plateau Escaping & Cycle Prevention**:
   - Local search algorithms can get trapped on local plateaus where no single move strictly decreases conflicts. Min-Conflicts employs three complementary plateau-escaping mechanisms:
     * **Random Walk**: With probability $p_{\text{walk}} = 0.05$, the solver selects a random legal placement from the variable's domain instead of a greedy minimum-conflict placement.
     * **Stagnation Perturbation**: If the best conflict count fails to improve for `stagnation_limit = 150` consecutive iterations, the solver forces a random perturbation on a conflicted variable.
     * **Non-stationary Preference**: When multiple placements achieve the same minimum conflict score, the solver preferentially chooses an alternative placement over the current placement to avoid zero-progress stationary steps.

5. **Stop Authority**:
   - When all conflicts reach 0, the final schedule is verified with `validate_timetable(tt)` from `core.constraints`. A schedule is only marked `solved` if `validate_timetable` confirms `is_complete == True` and `hard_violation_count == 0`.

### 8.2 Min-Conflicts Metrics Definition
- **`iterations`**: Number of local repair variable reassignment steps (NOT backtracking tree nodes).
- **`conflicted_variable_selections`**: Number of times a conflicted variable was chosen for move evaluation.
- **`candidate_placements_evaluated`**: Total number of candidate placement moves scored against occupancy tables.
- **`initial_hard_conflicts`**: Number of pairwise clashes in the initial complete assignment.
- **`final_hard_violations`**: Hard violations in the final timetable according to `validate_timetable`.
- **`random_seed`**: Pseudo-random seed used for reproducible experiments.

### 8.3 Benchmark Results (10 Runs with Different Seeds)

Because Min-Conflicts is a randomized local search algorithm, single runs cannot be compared directly with deterministic algorithms. Below are the actual measurements across 10 randomized trials (seeds 1 to 10) using random initialization (`initial_strategy="random"`):

#### Small Academic Dataset (10 Runs, Seeds 1–10)
| Seed | Status | Runtime | Iterations | Placements Scored | Initial Conflicts | Hard Violations | Soft Penalties (S1 / S2 / S3) | Total Cost |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `SOLVED` | 0.0018s | 14 | 785 | 29 | 0 | 9 / 9 / 0 | 18 |
| 2 | `SOLVED` | 0.0011s | 14 | 763 | 32 | 0 | 10 / 8 / 0 | 18 |
| 3 | `SOLVED` | 0.0019s | 23 | 916 | 30 | 0 | 10 / 10 / 0 | 20 |
| 4 | `SOLVED` | 0.0014s | 20 | 937 | 38 | 0 | 5 / 9 / 0 | **14** |
| 5 | `SOLVED` | 0.0017s | 21 | 914 | 30 | 0 | 10 / 13 / 0 | 23 |
| 6 | `SOLVED` | 0.0014s | 14 | 716 | 29 | 0 | 15 / 7 / 0 | 22 |
| 7 | `SOLVED` | 0.0014s | 14 | 762 | 27 | 0 | 8 / 10 / 0 | 18 |
| 8 | `SOLVED` | 0.0010s | 13 | 638 | 24 | 0 | 11 / 8 / 0 | 19 |
| 9 | `SOLVED` | 0.0012s | 17 | 763 | 29 | 0 | 11 / 5 / 0 | 16 |
| 10 | `SOLVED` | 0.0011s | 13 | 658 | 21 | 0 | 9 / 9 / 0 | 18 |
| **Avg** | **100% Success** | **0.0014s** | **16.3** | **785.2** | **28.9** | **0** | **9.8 / 8.8 / 0.0** | **18.6** |

#### Large Academic Dataset (10 Runs, Seeds 1–10)
| Seed | Status | Runtime | Iterations | Placements Scored | Initial Conflicts | Hard Violations | Soft Penalties (S1 / S2 / S3) | Total Cost |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `SOLVED` | 0.0071s | 40 | 4,979 | 65 | 0 | 35 / 11 / 0 | **46** |
| 2 | `SOLVED` | 0.0112s | 52 | 6,052 | 101 | 0 | 39 / 16 / 0 | 55 |
| 3 | `SOLVED` | 0.0088s | 51 | 5,377 | 65 | 0 | 37 / 16 / 0 | 53 |
| 4 | `SOLVED` | 0.0086s | 48 | 5,923 | 88 | 0 | 38 / 25 / 0 | 63 |
| 5 | `SOLVED` | 0.0071s | 43 | 5,141 | 73 | 0 | 38 / 13 / 0 | 51 |
| 6 | `SOLVED` | 0.0082s | 46 | 5,707 | 75 | 0 | 30 / 19 / 0 | 49 |
| 7 | `SOLVED` | 0.0080s | 47 | 5,492 | 87 | 0 | 32 / 18 / 0 | 50 |
| 8 | `SOLVED` | 0.0075s | 46 | 5,576 | 76 | 0 | 36 / 17 / 0 | 53 |
| 9 | `SOLVED` | 0.0061s | 38 | 4,763 | 62 | 0 | 57 / 12 / 0 | 69 |
| 10 | `SOLVED` | 0.0057s | 34 | 4,338 | 49 | 0 | 45 / 18 / 0 | 63 |
| **Avg** | **100% Success** | **0.0078s** | **44.5** | **5,334.8** | **74.1** | **0** | **38.7 / 16.5 / 0.0** | **55.2** |

### 8.4 Algorithm Comparison: Constructive Search vs. Local Search

| Dimension | Backtracking (Plain / MRV / FC) | Min-Conflicts Local Search |
|---|---|---|
| **Search Paradigm** | Systematic tree search (starts empty, assigns 1 variable at a time). | Stochastic local search (starts complete with conflicts, repairs iteratively). |
| **Completeness** | **Complete**: Guaranteed to find a solution if one exists, or prove unsatisfiability. | **Incomplete**: Does not guarantee finding a solution; cannot prove unsatisfiability. |
| **Speed on Feasible Instances** | Fast (0.0008s–0.067s depending on heuristics). | Extremely fast (0.0014s on small, 0.0078s on large). |
| **Over-constrained Problems** | Proves unsatisfiable when search space exhausts. | Runs until `max_iterations` expires without proof. |
| **Memory Footprint** | $O(N \cdot D)$ recursion stack + pruned domain states. | $O(N)$ assignment map + occupancy counters. |
| **Handling of Plateaus** | Backtracks up the tree when dead ends occur. | Requires random walk / stagnation perturbation to escape local minima. |

### 8.5 Running Min-Conflicts

#### CLI Runner (`run_solver.py`)
```bash
# Run Min-Conflicts on small dataset with default reproducible seed (42)
python run_solver.py --dataset small --algorithm min_conflicts
python run_solver.py --dataset small --mc

# Run on large dataset with custom seed and iteration limit
python run_solver.py --dataset large --mc --seed 123 --max-iterations 20000

# Run with greedy initial assignment strategy
python run_solver.py --dataset small --mc --initial-strategy greedy

# Save solved schedule to JSON file
python run_solver.py --dataset small --mc --output solution_mc.json
```

#### Python Script
```python
from pathlib import Path
from core.models import load_dataset_from_file
from core.min_conflicts import solve_min_conflicts

dataset = load_dataset_from_file(Path("datasets/small_dataset.json"))

result = solve_min_conflicts(
    dataset,
    max_iterations=10_000,
    random_seed=42,
    initial_strategy="random",
)

print(f"Status:     {result.status}")
print(f"Runtime:    {result.runtime_seconds:.4f}s")
print(f"Iterations: {result.iterations}")
print(f"Cost:       {result.total_cost}")
```

---

## 9. Streamlit Web Application (`app.py`)
A full-featured Streamlit application provides a visual interface to the scheduling engine.

### 9.1 Launch

```bash
streamlit run app.py
```

### 9.2 Features

The application has six pages accessible via the sidebar:

| Page | Description |
|---|---|
| **Overview & Dataset** | Select and inspect the small or large dataset. View classes, teachers, rooms, subjects, time structure, and feasibility checks. |
| **Generate Timetable** | Choose any of the 5 algorithms, configure solver parameters (max nodes, time limit, random seed, initial strategy), and generate a timetable. Displays success/failure status, runtime, search metrics, soft penalties, total cost, and cost-vs-iterations plot for Min-Conflicts. |
| **View Schedule** | Interactive grid view grouped by Class, Teacher, or Room. Multi-period labs show continuation markers. |
| **Edit Schedule** | Select a session, choose a new (day, period, room) placement from the unary-valid domain, and apply. Edits are validated against all hard constraints before acceptance. Invalid edits are rejected with violation details. Revert restores the original generated timetable. Cost is recalculated after every valid edit. |
| **Export** | Download the schedule as CSV or Excel. Columns: Class, Subject, Session ID, Teacher, Day, Start Period, Duration, Room, Type. |
| **Experiment Results** | Display the benchmark summary table, individual runs, and Min-Conflicts cost-vs-iterations plot from `experiments/results/`. |

### 9.3 Algorithm Configurations

| UI Label | Solver Call |
|---|---|
| Plain Backtracking | `solve_backtracking(ds, use_mrv=False, use_degree=False, use_fc=False)` |
| MRV Backtracking | `solve_backtracking(ds, use_mrv=True)` |
| MRV + Degree Heuristic | `solve_backtracking(ds, use_mrv=True, use_degree=True)` |
| MRV + Degree + Forward Checking | `solve_backtracking(ds, use_mrv=True, use_degree=True, use_fc=True)` |
| Min-Conflicts | `solve_min_conflicts(ds, max_iterations=N, random_seed=S)` |

---

## 10. Experimental Evaluation Framework

### 10.1 Running Experiments

```bash
python experiments/run_experiments.py
```

This runs all 5 algorithms on both datasets (with 10 randomized Min-Conflicts trials per dataset) and produces:

| Output File | Description |
|---|---|
| `experiments/results/experiment_runs.csv` | Per-run metrics for all 28 executions |
| `experiments/results/experiment_summary.csv` | Aggregated averages by config and dataset |
| `experiments/results/experiment_metadata.json` | Timestamps, library versions, run count |
| `experiments/results/min_conflicts_cost_vs_iterations.png` | Cost convergence plot |

### 10.2 Customisation

Edit `experiments/run_experiments.py` to change:
- Number of Min-Conflicts trials per dataset
- Max iterations and time limits
- Random seed ranges

---

## 11. Project Structure

```
Timetable/
├── core/
│   ├── __init__.py          # Package metadata & solver exports
│   ├── models.py            # Entity dataclasses, schema validation, feasibility checks
│   ├── sessions.py          # SessionVar generation, Placement, Timetable container
│   ├── constraints.py       # H1-H9 + STRUCT validators, validate_timetable
│   ├── cost.py              # S1-S3 soft penalties, CostBreakdown, calculate_cost
│   ├── backtracking.py      # CSP solver (Plain, MRV, MRV+Degree, MRV+Degree+FC)
│   └── min_conflicts.py     # Min-Conflicts Local Search solver
├── datasets/
│   ├── small_dataset.json
│   ├── large_dataset.json
│   ├── small_feasible_schedule.json
│   └── large_feasible_schedule.json
├── experiments/
│   ├── run_experiments.py   # Automated experiment runner
│   └── results/             # Generated CSV, JSON, and PNG outputs
├── tests/
│   ├── conftest.py          # Shared dataset fixtures
│   ├── test_models.py       # Data model loading and validation
│   ├── test_sessions.py     # Variable generation, placement, timetable lifecycle
│   ├── test_constraints.py  # Isolated H1-H9 violation tests
│   ├── test_cost.py         # Soft penalties and cost formula
│   ├── test_datasets.py     # Dataset structure and feasibility
│   ├── test_feasibility.py  # Hand-built schedule validation
│   ├── test_backtracking.py # Backtracking solver tests (51 tests)
│   ├── test_min_conflicts.py# Min-Conflicts tests (18 tests)
│   ├── test_experiments.py  # Experiment runner tests (32 tests)
│   └── test_app.py          # Streamlit app tests (55 tests)
├── app.py                   # Streamlit web application
├── run_solver.py            # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 12. Installation & Testing

### 12.1 Setup

Requires Python 3.10+.

```bash
cd d:/Timetable

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 12.2 Dependencies

`requirements.txt` contains:
- `pytest>=8.0.0` — test framework
- `streamlit>=1.30.0` — web application
- `pandas>=2.0.0` — data manipulation
- `matplotlib>=3.7.0` — plotting
- `openpyxl>=3.1.0` — Excel export

### 12.3 Run Tests

```bash
python -m pytest -v
```

Expected output:
```
=================== 201 passed in 4.38s ====================
```

### 12.4 Launch the Application

```bash
streamlit run app.py
```

### 12.5 Run Solvers from CLI

```bash
python run_solver.py --dataset small --algorithm plain
python run_solver.py --dataset small --algorithm mrv
python run_solver.py --dataset small --algorithm mrv_degree
python run_solver.py --dataset small --algorithm mrv_degree_fc
python run_solver.py --dataset small --algorithm min_conflicts
python run_solver.py --dataset large --mc --seed 42
```

### 12.6 Generate Experiment Results

```bash
python experiments/run_experiments.py
```

---

## 13. Viva Examination Quick-Reference Guide

When explaining this implementation to the examiner in your viva:

1. **Q: Why are multi-period labs treated as a single session with `duration = 2` instead of two separate sessions?**  
   *A*: Modeling a 2-hour lab as two separate 1-hour sessions introduces unnecessary binary constraints requiring them to be adjacent in time and assigned to the same room and teacher. By modeling it as a single variable with `duration = 2`, contiguity is guaranteed by construction, keeping the CSP variable space smaller and eliminating constraint thrashing.

2. **Q: What is the difference between unary and binary constraints in your validator?**  
   *A*: Unary constraints depend only on a single variable's assignment: H4 (Room capacity vs Class strength), H5 (Room type vs Session type), and H6 (Teacher availability vs Slot). Binary constraints depend on relationships between pairs of variables: H1 (Teacher conflict), H2 (Class conflict), and H3 (Room conflict).

3. **Q: Why does an empty timetable have non-zero hard violations?**  
   *A*: Because of constraint H7 (Weekly Syllabus Hours). H7 measures $|\text{assigned periods} - \text{required periods}|$. If a subject requires 4 hours and 0 are assigned, there are 4 missing period violations. This prevents incomplete or empty timetables from ever being marked valid.

4. **Q: How does the cost function prioritize hard constraints over soft constraints?**  
   *A*: The hard violation multiplier is set to 1000 ($1000 \times \text{hard violations}$). Even if a timetable has several soft penalties (e.g. 5 gaps $\times 1 = 5$), a single hard violation ($1000$) immediately incurs a far greater penalty, strictly directing local search or heuristics away from infeasible states.

5. **Q: What is Forward Checking and how does it differ from Plain Backtracking and MRV?**  
   *A*: Plain Backtracking tests constraints only when assigning a variable. Backtracking + MRV counts remaining legal values dynamically before selecting a variable, but re-evaluates them on the fly. Forward Checking maintains active variable domains persistently: whenever variable $X$ is assigned value $v$, it immediately prunes conflicting values from the domains of all unassigned neighbors. If any neighbor's domain becomes empty (domain wipeout), the assignment is immediately rejected without searching deeper.

6. **Q: Why did Forward Checking speed up MRV search so significantly?**  
   *A*: Without Forward Checking, MRV must iterate through all valid placements of every unassigned variable at every node to check dynamic collision constraints, costing $O(|V_{\text{unassigned}}| \times |D|)$ checks per node. With Forward Checking, legal domains are already maintained in `current_domains`, allowing $O(1)$ length lookups (`len(current_domains[v])`).

7. **Q: Does Forward Checking guarantee global consistency?**  
   *A*: No. Forward Checking only ensures arc consistency between the *assigned* variable and its unassigned neighbors (1-step lookahead). It does not detect inconsistencies that exist between pairs of *unassigned* variables. To detect two unassigned variables whose remaining values conflict with each other before making an assignment, full Arc Consistency (e.g. AC-3 or Maintaining Arc Consistency - MAC) would be required.

8. **Q: What is Min-Conflicts and how does it differ fundamentally from Backtracking?**  
   *A*: Backtracking is a systematic, constructive search algorithm that builds a partial schedule variable-by-variable, exploring a depth-first search tree and pruning branches when constraints fail. Min-Conflicts is a local search heuristic that starts from a **complete schedule** (which initially has conflicts) and iteratively repairs clashes by selecting a conflicted variable and reassigning it to the value that minimizes conflicting relationships.

9. **Q: Can Min-Conflicts prove that a timetable problem is unsatisfiable?**  
   *A*: No. Min-Conflicts is an **incomplete** local search algorithm. If a timetable instance has no mathematically feasible schedule, Min-Conflicts will simply run until `max_iterations` expires and return `limit_reached`. To prove unsatisfiability, a complete search algorithm such as Backtracking (or Backtracking + FC) is required.

10. **Q: How does Min-Conflicts avoid getting stuck in local minima or plateaus?**  
    *A*: It employs three mechanisms: (1) Random walk with probability $p_{\text{walk}} = 0.05$ to explore non-greedy moves; (2) Stagnation detection that forces a random perturbation if the best conflict count does not improve for 150 consecutive steps; and (3) Non-stationary tie-breaking that prefers an alternative minimum-conflict placement over remaining stationary on flat plateaus.

11. **Q: How does the Streamlit manual edit feature work?**  
    *A*: The user selects a placed session and a new (day, period, room) from the unary-valid domain. The candidate timetable is validated against all H1-H9 constraints. If any violations exist, the edit is rejected with the specific violations displayed. Valid edits are applied and the cost is immediately recalculated and shown.
