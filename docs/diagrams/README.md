# Technical Architecture & Algorithm Diagrams

This directory contains technical diagrams illustrating the architecture, workflows, CSP modeling, search algorithms, and constraint evaluation mechanisms implemented in the **AI-Based Timetable Scheduler**.

Each diagram is provided as a standalone [Mermaid](https://mermaid.js.org/) source file (`.mmd`) and rendered below for review on GitHub.

---

## Diagram Index

1. [Diagram A: System Architecture](#diagram-a-system-architecture) (`system_architecture.mmd`)
2. [Diagram B: Timetable Generation Workflow](#diagram-b-timetable-generation-workflow) (`timetable_generation_workflow.mmd`)
3. [Diagram C: CSP Problem Formulation](#diagram-c-csp-problem-formulation) (`csp_formulation.mmd`)
4. [Diagram D: Systematic Backtracking CSP Solvers](#diagram-d-systematic-backtracking-csp-solvers) (`backtracking_csp.mmd`)
5. [Diagram E: Min-Conflicts Local Search](#diagram-e-min-conflicts-local-search) (`min_conflicts_algorithm.mmd`)
6. [Diagram F: Hard & Soft Constraint Evaluation](#diagram-f-hard--soft-constraint-evaluation) (`constraint_evaluation.mmd`)
7. [Diagram G: Data Import & Custom Timetable Setup](#diagram-g-data-import--custom-timetable-setup) (`data_import_setup.mmd`)

---

### Diagram A: System Architecture

Source: [`system_architecture.mmd`](system_architecture.mmd)

```mermaid
flowchart TB
    subgraph UI ["Presentation Layer (Streamlit app.py)"]
        direction TB
        DASH["📊 Dashboard<br/>Active status & history preview"]
        CREATE["🛠️ Create Timetable<br/>Wizard, Table Editor, Matrix"]
        MYTT["📁 My Timetables<br/>Persistent JSON storage manager"]
        VIEW["👁️ View & Edit Timetable<br/>Class/Teacher/Room grids & clash editor"]
        EXP["🔬 Experiments & Benchmarks<br/>Summary tables & convergence plots"]
        EXPORT["📥 Export<br/>CSV, Excel (.xlsx), Dataset JSON"]
    end

    subgraph INGEST ["Data Ingestion & Building Layer"]
        BI["core/bulk_importer.py<br/>TSV paste, CSV/XLSX parser,<br/>alias normalizer, templates"]
        CB["core/custom_builder.py<br/>Relational validator,<br/>slot parser, dataset builder"]
    end

    subgraph CORE_MODELS ["Core Data Models & CSP State"]
        DS["core/models.py<br/>TimetableDataset, ClassGroup,<br/>Teacher, Subject, Room, TimeStructure"]
        SESS["core/sessions.py<br/>SessionVar (Atomic Theory dur=1, Lab dur=2)<br/>Placement (day, period, room)<br/>Timetable assignment container"]
    end

    subgraph SOLVERS ["AI Solver Engine"]
        direction TB
        subgraph BT_MODULE ["core/backtracking.py (Systematic Search)"]
            BT_PLAIN["Plain Backtracking (DFS Baseline)"]
            BT_MRV["MRV Heuristic (Min Remaining Values)"]
            BT_DEG["Degree Heuristic (Tie-Breaker)"]
            BT_FC["Forward Checking (Domain Lookahead & Wipeout Pruning)"]
        end
        subgraph MC_MODULE ["core/min_conflicts.py (Heuristic Local Search)"]
            MC_BASE["Min-Conflicts Repair Loop<br/>Exact H1-H3 Clash Scoring"]
            MC_ESC["Plateau Escaping<br/>Random Walk (p=0.05) & Stagnation Perturbation"]
            MC_REF["Optional Two-Stage Soft Refinement"]
        end
        subgraph SA_MODULE ["core/simulated_annealing.py (Hybrid Optimization)"]
            SA_P1["Phase 1: Backtracking MRV+Degree+FC Feasible Seed (H=0)"]
            SA_P2["Phase 2: Metropolis Simulated Annealing<br/>Safe relocations & swaps minimizing soft cost"]
        end
    end

    subgraph EVAL ["Constraint & Cost Evaluation Layer"]
        VAL["core/constraints.py<br/>Hard Validators: H1–H9 + STRUCT<br/>Independent validate_timetable()"]
        COST["core/cost.py<br/>Soft Evaluators: S1–S5<br/>calculate_cost() (1000*Hard + Weighted Soft)"]
    end

    subgraph STORAGE_EXP ["Storage & Experiment Execution"]
        STORE["core/storage.py (TimetableStore)<br/>Atomic JSON persistence in saved_timetables/"]
        RUN_EXP["experiments/run_experiments.py<br/>Standardized benchmarks, master seeds,<br/>CSV summaries & Matplotlib curves"]
    end

    %% Connections
    CREATE --> BI
    CREATE --> CB
    BI --> CB
    CB --> DS
    DS --> SESS

    SESS --> SOLVERS
    SOLVERS --> EVAL
    EVAL --> SESS

    SOLVERS --> VIEW
    SESS --> VIEW
    VIEW --> VAL
    VIEW --> COST
    VIEW --> STORE

    STORE --> MYTT
    STORE --> DASH
    SESS --> EXPORT
    RUN_EXP --> EXP
    RUN_EXP --> SOLVERS
```

---

### Diagram B: Timetable Generation Workflow

Source: [`timetable_generation_workflow.mmd`](timetable_generation_workflow.mmd)

```mermaid
flowchart TD
    START([User Initiates Timetable Creation]) --> CHOICE{Input Method}

    CHOICE -->|Predefined Benchmark| BENCH[Load small, medium_tight,<br/>or large dataset.json]
    CHOICE -->|Quick Setup Wizard / Tables| MANUAL[Interactive Form Input<br/>Grid, Classes, Teachers, Rooms]
    CHOICE -->|Bulk Ingestion| BULK[TSV Clipboard Paste,<br/>CSV, or XLSX Workbook]

    BULK --> NORM[Header Normalization<br/>Alias Mapping to Canonical Schema]
    NORM --> ROW_VAL{Row Validation}
    ROW_VAL -->|Invalid Rows| ERR_ROW[Display Detailed Row Errors<br/>Preserve valid rows for correction]
    ERR_ROW --> BULK

    ROW_VAL -->|Valid Rows| BUILD_DS
    MANUAL --> BUILD_DS[custom_builder.py<br/>Construct TimetableDataset]
    BENCH --> BUILD_DS

    BUILD_DS --> FEAS_CHECK{Pre-Solve Feasibility & Schema Checks}
    FEAS_CHECK -->|Infeasible / Invalid Schema| FEAS_FAIL[Display Validation Errors<br/>Room capacity, total hours vs slot budget]
    FEAS_FAIL --> CHOICE

    FEAS_CHECK -->|Feasible & Valid| GEN_VARS[sessions.py<br/>Generate Atomic Session Variables<br/>Theory: dur=1, Lab: dur=2]
    GEN_VARS --> FILTER_DOM[Pre-filter Unary Domains<br/>Room type, room cap, teacher unavail,<br/>class lunch, blocked slots]

    FILTER_DOM --> ALGO_SELECT{Select AI Solver}

    ALGO_SELECT -->|Systematic| BT[core/backtracking.py<br/>Plain / MRV / MRV+Degree / MRV+Degree+FC]
    ALGO_SELECT -->|Local Search| MC[core/min_conflicts.py<br/>Min-Conflicts Repair Loop<br/>Random Walk & Stagnation Escape]
    ALGO_SELECT -->|Hybrid| SA[core/simulated_annealing.py<br/>Phase 1: MRV+Degree+FC Feasible Seed<br/>Phase 2: Metropolis Soft Cost Annealing]

    BT --> SOLVE_EXEC{Solver Execution}
    MC --> SOLVE_EXEC
    SA --> SOLVE_EXEC

    SOLVE_EXEC -->|Budget Exceeded / Unsatisfiable| FAIL_RES[Report Failure Reason<br/>Nodes explored, time elapsed]
    FAIL_RES --> RETRY{Adjust Settings?}
    RETRY -->|Yes| ALGO_SELECT
    RETRY -->|No| STOP([Aborted])

    SOLVE_EXEC -->|Solution Found| INDEP_VAL[Independent Verification<br/>validate_timetable H1–H9 + STRUCT]
    INDEP_VAL --> COST_EVAL[Soft Cost Evaluation<br/>calculate_cost S1–S5 Breakdown]

    COST_EVAL --> ACTIVE[Set Active Timetable in Session State<br/>Display Summary Banner & Metrics]

    ACTIVE --> POST_ACTION{User Action}
    POST_ACTION -->|Inspect| GRIDS[View Timetable Grids<br/>By Class, Teacher, or Room]
    POST_ACTION -->|Edit| MANUAL_EDIT[Manual Schedule Editor<br/>Move session slot/room with real-time clash check]
    POST_ACTION -->|Persist| SAVE[Save to My Timetables<br/>Atomic JSON in saved_timetables/]
    POST_ACTION -->|Export| EXPORT[Export to CSV, Excel .xlsx, or Dataset JSON]

    MANUAL_EDIT --> INDEP_VAL
```

---

### Diagram C: CSP Problem Formulation

Source: [`csp_formulation.mmd`](csp_formulation.mmd)

```mermaid
flowchart LR
    subgraph VARS ["Variables (X) — SessionVar"]
        direction TB
        V1["<b>Theory SessionVar</b><br/>id: 'C1_MATH_1'<br/>class_id: 'C1'<br/>subject_id: 'MATH'<br/>teacher_id: 'T1'<br/>duration: 1 period<br/>session_type: 'theory'"]
        V2["<b>Lab SessionVar (Atomic)</b><br/>id: 'C1_PHYLAB_1'<br/>class_id: 'C1'<br/>subject_id: 'PHYLAB'<br/>teacher_id: 'T2'<br/>duration: 2 consecutive periods<br/>session_type: 'lab'"]
    end

    subgraph DOMAINS ["Domains (D) — Placement"]
        direction TB
        PLC["<b>Placement Coordinates</b><br/>(day, start_period, room_id)<br/>Occupies periods:<br/>{p, ..., p + duration - 1}"]
        UNARY["<b>Unary Pruning Filters</b><br/>• H5 Room Type (Lab in lab, theory in classroom)<br/>• H4 Capacity (room.capacity >= class.strength)<br/>• H6 Availability ((day, p) not in teacher.unavailable_slots)<br/>• H6 Blocked Slots ((day, p) not in blocked_periods)<br/>• H9 Lunch Break (p != class.lunch_slot)<br/>• STRUCT Bounds (p + duration <= periods_per_day)"]
        PLC --- UNARY
    end

    subgraph CONSTRAINTS ["Constraints (C) — Hard & Soft"]
        direction TB
        subgraph BINARY ["Binary Clashes (Evaluated per overlapping period)"]
            H1["<b>H1 Teacher Clash</b><br/>Teacher cannot be double-booked<br/>slots(S1) ∩ slots(S2) = ∅ if teacher(S1) == teacher(S2)"]
            H2["<b>H2 Class Clash</b><br/>Class batch cannot attend 2 sessions<br/>slots(S1) ∩ slots(S2) = ∅ if class(S1) == class(S2)"]
            H3["<b>H3 Room Clash</b><br/>Room cannot host 2 sessions<br/>slots(S1) ∩ slots(S2) = ∅ if room(S1) == room(S2)"]
        end
        subgraph GLOBAL ["Global & Quota Constraints"]
            H7["<b>H7 Weekly Syllabus Hours</b><br/>|assigned_periods - required_periods| == 0"]
            H8["<b>H8 Teacher Max Load</b><br/>assigned_periods <= teacher.max_weekly_load"]
        end
        subgraph SOFT ["Soft Quality Metrics (Weights w1-w5)"]
            S1["S1: Class Day Idle Gaps (w1=1)"]
            S2["S2: Subject Daily Repeats (w2=1)"]
            S3["S3: Lab Inter-Session Continuity (w3=1)"]
            S4["S4: Teacher Slot Preferences (w4=1)"]
            S5["S5: Teacher Max Consec Lectures >2 (w5=2)"]
        end
    end

    VARS -->|Assign Placement| DOMAINS
    DOMAINS -->|Subject to| CONSTRAINTS
    CONSTRAINTS --> OBJECTIVE["<b>Objective Cost Evaluation</b><br/>Total Cost = 1000 × (H1..H9 + STRUCT) + Σ (w_i × S_i)"]
```

---

### Diagram D: Systematic Backtracking CSP Solvers

Source: [`backtracking_csp.mmd`](backtracking_csp.mmd)

```mermaid
flowchart TD
    START([solve_backtracking]) --> VAR_CHECK{All variables assigned?}
    VAR_CHECK -->|Yes: Assignment Complete| VERIFY[validate_timetable H1–H9 + STRUCT<br/>Confirm 0 hard violations]
    VERIFY --> SOLVED([Return Status: SOLVED<br/>Timetable Found])

    VAR_CHECK -->|No: Unassigned Exist| BUDGET{Node or Time Limit Exceeded?}
    BUDGET -->|Yes| LIMIT([Return Status: LIMIT_REACHED])

    BUDGET -->|No| SELECT_VAR{Select Next Variable}

    subgraph VAR_STRATEGY ["Variable Selection Strategies"]
        direction TB
        PLAIN_SEL["<b>Plain Baseline</b><br/>Static sequential index order"]
        MRV_SEL["<b>MRV (Minimum Remaining Values)</b><br/>Variable with fewest currently legal placements"]
        DEG_SEL["<b>MRV + Degree Heuristic</b><br/>MRV with tie-breaker: variable with most<br/>binary constraints on unassigned variables"]
    end

    SELECT_VAR -.-> PLAIN_SEL
    SELECT_VAR -.-> MRV_SEL
    SELECT_VAR -.-> DEG_SEL
    PLAIN_SEL --> ITER_DOM
    MRV_SEL --> ITER_DOM
    DEG_SEL --> ITER_DOM

    ITER_DOM[Get Candidate Placements for Variable<br/>Standard domain order or Soft-cost biased] --> DOM_LOOP{More placements in domain?}

    DOM_LOOP -->|No: Domain Exhausted| BACKTRACK[Backtrack:<br/>Unassign current variable<br/>Restore any pruned domains<br/>Increment backtrack counter]
    BACKTRACK --> RETURN_FAIL[Return False to caller]

    DOM_LOOP -->|Yes: Try Next Placement| TRY_PLC[Assign Variable = Placement<br/>Increment assignments_tried]

    TRY_PLC --> CHECK_CLASH{Incremental Clash Check<br/>H1: Teacher double-booked?<br/>H2: Class double-booked?<br/>H3: Room double-booked?}

    CHECK_CLASH -->|Collision Detected| UNDO_LOCAL[Reject Placement] --> DOM_LOOP

    CHECK_CLASH -->|No Collision| FC_CHECK{Forward Checking Enabled?}

    FC_CHECK -->|No: Plain / MRV / Degree| RECURSE[Recurse: solve next variable<br/>nodes_explored++]
    
    FC_CHECK -->|Yes: mrv_degree_fc| PRUNE[Prune Placements from Active Domains<br/>of all unassigned constraint neighbors]
    PRUNE --> WIPEOUT{Domain Wipeout?<br/>Any neighbor domain size == 0?}
    
    WIPEOUT -->|Yes: Wipeout Detected| RESTORE_DOM[Fail Fast!<br/>Restore pruned neighbour domains] --> UNDO_LOCAL

    WIPEOUT -->|No: Domains Healthy| RECURSE

    RECURSE --> REC_RESULT{Recursive Call Succeeded?}
    REC_RESULT -->|Yes| SOLVED
    REC_RESULT -->|No: Dead End| RESTORE_FC[Restore neighbour pruned domains<br/>if Forward Checking active] --> BACKTRACK
```

---

### Diagram E: Min-Conflicts Local Search

Source: [`min_conflicts_algorithm.mmd`](min_conflicts_algorithm.mmd)

```mermaid
flowchart TD
    START([solve_min_conflicts]) --> INIT[Generate Complete Initial Assignment<br/>Random or Greedy from valid unary domains]
    INIT --> CHECK_DOM{Any variable has empty domain?}
    CHECK_DOM -->|Yes| UNSAT([Return Status: UNSATISFIABLE])

    CHECK_DOM -->|No: Valid Complete State| ASSESS[Assess Binary Conflicts for all variables<br/>H1 Teacher, H2 Class, H3 Room Clashes]

    ASSESS --> CONF_CHECK{Conflicted Variables Count == 0?}
    CONF_CHECK -->|Yes: 0 Hard Conflicts| INDEP_VAL[Independent Verification<br/>validate_timetable H1–H9 + STRUCT]
    
    INDEP_VAL --> TWO_STAGE{Two-Stage Soft Refinement Enabled?}
    TWO_STAGE -->|No| SOLVED([Return Status: SOLVED<br/>Conflict-Free Timetable])
    TWO_STAGE -->|Yes| SOFT_REFINE[Execute Local Moves Strictly Preserving 0 Hard Clashes<br/>Minimizing S1–S5 Soft Penalty Breakdown] --> SOLVED

    CONF_CHECK -->|No: Conflicts Exist| ITER_CHECK{Iteration Limit (5000)<br/>or Time Limit Exceeded?}
    ITER_CHECK -->|Yes| LIMIT([Return Status: LIMIT_REACHED<br/>Return Best Timetable Found])

    ITER_CHECK -->|No: Search Continues| STAGNATION{Stagnation Check:<br/>Global min conflict unimproved for 150 steps?}
    STAGNATION -->|Yes: Plateau Trapped| PERTURB[Force Random Perturbation on a Conflicted Variable<br/>Reset Stagnation Counter] --> EVAL_NEXT
    
    STAGNATION -->|No: Progress Normal| EVAL_NEXT[Select Conflicted Variable V<br/>Randomly from variables with conflict > 0]

    EVAL_NEXT --> WALK_CHECK{Random Walk Check?<br/>p < random_walk_probability (0.05)}
    
    WALK_CHECK -->|Yes: Explore| CHOOSE_RANDOM[Pick Random Legal Placement from Domain(V)] --> APPLY_MOVE
    
    WALK_CHECK -->|No: Greedy Min-Conflict| EVAL_ALL[Score All Candidate Placements in Domain(V)<br/>Count hypothetical H1–H3 collisions with other variables<br/>Self-exclusion: exclude V from conflict count]
    
    EVAL_ALL --> TIE_BREAK{Identify Placements with Min Conflict Score<br/>Apply Non-Stationary Tie-Breaking}
    TIE_BREAK -->|Ties Exist| PREFER_ALT[Prefer Moving to Alternative Placement<br/>over staying stationary in current slot] --> APPLY_MOVE
    TIE_BREAK -->|Unique Min| CHOOSE_MIN[Select Minimum-Conflict Placement] --> APPLY_MOVE

    APPLY_MOVE[Reassign Variable V = Chosen Placement<br/>iterations++, update cost history] --> ASSESS
```

---

### Diagram F: Hard & Soft Constraint Evaluation

Source: [`constraint_evaluation.mmd`](constraint_evaluation.mmd)

```mermaid
flowchart TD
    TIMETABLE[Timetable State: Placed Sessions & Unassigned Variables] --> SPLIT{Constraint Evaluation}

    subgraph HARD_TIER ["Hard Constraints (H1–H9 + STRUCT) — Feasibility Boundary"]
        direction TB
        H1["<b>H1 Teacher Clash</b>: 1 per conflicting pair of sessions per overlapping period"]
        H2["<b>H2 Class Clash</b>: 1 per conflicting pair of sessions per overlapping period"]
        H3["<b>H3 Room Clash</b>: 1 per conflicting pair of sessions per overlapping period"]
        H4["<b>H4 Room Capacity</b>: 1 per session where room.capacity < class.strength"]
        H5["<b>H5 Room Type</b>: 1 per session in incompatible room (theory in lab / lab in classroom)"]
        H6["<b>H6 Availability & Blocked</b>: 1 per occupied slot on teacher unavailable or global blocked slot"]
        H7["<b>H7 Weekly Syllabus Hours</b>: |assigned_periods - required_periods| per (class, subject)<br/><i>Unassigned sessions automatically count as missing periods!</i>"]
        H8["<b>H8 Teacher Max Load</b>: max(0, assigned_periods - teacher.max_weekly_load) per teacher"]
        H9["<b>H9 Class Lunch Break</b>: 1 per occupied slot overlapping class.lunch_slot"]
        STRUCT["<b>STRUCT Structural Validity</b>: 1 per out-of-bounds slot, overflow period, or invalid teacher-subject assignment"]

        HARD_SUM["Sum Total Hard Violations<br/><code>hard_violation_count</code>"]
        H1 --> HARD_SUM
        H2 --> HARD_SUM
        H3 --> HARD_SUM
        H4 --> HARD_SUM
        H5 --> HARD_SUM
        H6 --> HARD_SUM
        H7 --> HARD_SUM
        H8 --> HARD_SUM
        H9 --> HARD_SUM
        STRUCT --> HARD_SUM
        HARD_PENALTY["<b>Hard Penalty Cost</b><br/><code>1000 × hard_violation_count</code><br/><i>Dominates cost function ensuring feasible > infeasible</i>"]
        HARD_SUM --> HARD_PENALTY
    end

    subgraph SOFT_TIER ["Soft Constraints (S1–S5) — Pedagogical Quality Metrics"]
        direction TB
        S1["<b>S1 Class Daily Gaps</b> (w1 = 1.0)<br/>Idle periods between first and last lecture of class-day<br/><i>Excludes designated lunch break & blocked periods</i>"]
        S2["<b>S2 Subject Daily Spreading</b> (w2 = 1.0)<br/>Penalizes (count - 1) for duplicate theory lectures on same day<br/><i>Atomic lab sessions are strictly exempt</i>"]
        S3["<b>S3 Consecutive Labs Continuity</b> (w3 = 1.0)<br/>Atomic lab periods are consecutive by design;<br/>penalizes disjoint gap periods between separate lab blocks of same subject on same day"]
        S4["<b>S4 Teacher Slot Preferences</b> (w4 = 1.0)<br/>1 penalty per occupied period outside teacher's preferred_slots<br/><i>0 penalty if teacher specified no preferences</i>"]
        S5["<b>S5 Teacher Consecutive Hours</b> (w5 = 2.0)<br/>Penalizes runs of >2 consecutive teaching periods on same day<br/>Penalty = (streak_length - 2) × w5. Lunch/idle breaks streak."]

        SOFT_SUM["<b>Weighted Soft Cost</b><br/><code>w1·S1 + w2·S2 + w3·S3 + w4·S4 + w5·S5</code>"]
        S1 --> SOFT_SUM
        S2 --> SOFT_SUM
        S3 --> SOFT_SUM
        S4 --> SOFT_SUM
        S5 --> SOFT_SUM
    end

    SPLIT --> HARD_TIER
    SPLIT --> SOFT_TIER

    HARD_PENALTY --> TOTAL_COST["<b>Total Objective Cost</b><br/><code>total_cost = hard_cost + total_soft_cost</code><br/>Lower cost = Higher Timetable Quality"]
    SOFT_SUM --> TOTAL_COST

    TOTAL_COST --> STATUS_OUT["<b>Solution Status</b><br/>• is_valid: True iff complete & hard_violations == 0<br/>• is_complete: True iff 0 unassigned sessions"]
```

---

### Diagram G: Data Import & Custom Timetable Setup

Source: [`data_import_setup.mmd`](data_import_setup.mmd)

```mermaid
flowchart TD
    subgraph INPUT_SOURCES ["User Input Channels"]
        WIZARD["<b>Quick Setup Wizard</b><br/>5 Sequential Steps<br/>1. Grid & Breaks<br/>2. Classes & Subjects<br/>3. Teachers & Matrix<br/>4. Rooms & Capacities<br/>5. Solve & Verify"]
        TABLE_ED["<b>Classic Table Editor</b><br/>Direct tabular dataframes<br/>Classes, Teachers,<br/>Subjects, Rooms"]
        TSV_PASTE["<b>Excel / Sheets Clipboard Paste</b><br/>Direct TSV text paste<br/>into Streamlit textarea"]
        FILE_UPLOAD["<b>File Upload Ingestion</b><br/>• CSV files per entity<br/>• Single-table XLSX<br/>• Complete Multi-Sheet Workbook (.xlsx)"]
        MATRIX_UI["<b>Interactive Availability Matrix</b><br/>Visual Streamlit click grid<br/>Two-way sync with slot strings"]
        TEMPLATES["<b>Download Starter Templates</b><br/>Blank/Sample CSV, XLSX,<br/>or Combined Workbook"]
    end

    subgraph BULK_ENGINE ["Processing Engine (core/bulk_importer.py)"]
        PARSE["Delimiter Sniffing & Parsing<br/>TSV (tab), CSV (comma), Excel Sheet Reader"]
        NORM["Column Alias Normalization<br/>Maps arbitrary user headers<br/>(e.g., 'code', 'hours', 'cap', 'faculty')<br/>to Canonical Schema Columns"]
        FALLBACK["Default Fallback Injection<br/>Injects sensible defaults for optional fields<br/>(e.g., strength=30, lunch=4, load=14)"]
        ROW_VAL["Row-Level Validation (validate_bulk_rows)<br/>Generates accepted DataFrame<br/>Records RowError objects with line numbers"]
    end

    subgraph BUILDER_ENGINE ["Institutional Verification (core/custom_builder.py)"]
        SLOT_PARSER["Slot Coordinate Parsing (parse_slot_string)<br/>Converts 'Mon P1, Tue P3' to [(0,0), (1,2)]"]
        SYNTAX_VAL["Syntax & Field Checks<br/>Positive durations, non-empty IDs"]
        REL_VAL["Relational Integrity Checks<br/>• Teacher qualification for assigned subject<br/>• Class subject requirements reference valid subjects<br/>• Room capacity vs class strength"]
        FEAS_CHECK["Mathematical Feasibility (check_dataset_feasibility)<br/>• Total class hours <= Total room-slots<br/>• Lab hours <= Lab room-slots"]
    end

    TEMPLATES -.->|User downloads & fills| FILE_UPLOAD
    TSV_PASTE --> PARSE
    FILE_UPLOAD --> PARSE
    PARSE --> NORM
    NORM --> FALLBACK
    FALLBACK --> ROW_VAL

    ROW_VAL -->|Rejected Rows Detected| DIAG_ERR[Display Row-Level Errors in UI<br/>Keep valid rows intact for editing]
    DIAG_ERR --> TABLE_ED

    ROW_VAL -->|Accepted Clean Rows| MERGE[Merge into Custom Tables]
    WIZARD --> MERGE
    TABLE_ED --> MERGE
    MATRIX_UI --> SLOT_PARSER
    SLOT_PARSER --> MERGE

    MERGE --> SYNTAX_VAL
    SYNTAX_VAL --> REL_VAL
    REL_VAL --> FEAS_CHECK

    FEAS_CHECK -->|Violations Found| UI_WARN[Display Actionable Capacity / Integrity Alerts]
    UI_WARN --> TABLE_ED

    FEAS_CHECK -->|Checks Pass| DATASET[Construct TimetableDataset<br/>Ready for AI Solver Execution]
```
