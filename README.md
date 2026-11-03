# AutoSciAgent

We aim to develop a secure, self-verifying scientific computing workflow working as an autonomous, general-purpose **1D/2D/3D Physical Simulation and Partial Differential Equation (PDE) Solver Agent** (**AutoSciAgent**), based on **Hierarchical Feedback Control Theory** and **Agent = Model + Harness** ([HARNESS²](https://arxiv.org/abs/2610.14224) and [RRSI](https://arxiv.org/abs/2609.24972)).

Given a high-level physical simulation task (`tasks/<task_id>/<task_id>.md`), optional parameter text files (`tasks/<task_id>/*.txt`), and markdown harness instructions (`harness/*.md`), the agent processes tasks sequentially one by one, and autonomously:

- **Parses requirements, extracts physical/numerical parameters into variables from `.md`/`.txt` files, and runs a Pre-Audit Feasibility Gate** directly on the decoupled, read-only `tasks/<task_id>/` folder—evaluating 1D/2D/3D PDE well-posedness and clamping grid memory within safe CPU/GPU RAM bounds before planning.
- **Checks prior numerical knowledge in strict priority order (`SKILL.md` $\rightarrow$ `library/INDEX.md` & `EXPERIENCE.md`) and skips them if empty**, falling back cleanly to first-principles mathematical derivation.
- **Derives numerical discretizations and audits the math before coding**—verifying 1D/2D/3D Finite Difference / Finite Volume stencils, variable-coefficient interface fluxes, Dirichlet/Neumann/Robin boundary conditions, and Courant–Friedrichs–Lewy (CFL) stability criteria via an internal **Derivation Audit Gate**.
- **Writes variable-driven, CPU/GPU-adaptive Python solvers and debugs using execution logs and intermediate field digests**—reading all domain, mesh, boundary, and scenario parameters from `specs/simulation_spec.json` into variables (zero hardcoded constants) and inspecting full tracebacks, `.npz` array statistics/boundary slices (`inspect_results.py`), and `.csv` tables to self-correct bugs without dead loops.
- **Executes solvers autonomously inside an isolated per-task workspace** (`workspace/<task_id>/`) using a shared project virtual environment (`.venv/`) with clean-slate Copy-on-Write output archiving, configurable timeouts, deterministic spec-aware output verification (`verify_outputs.py`), and runtime middleware hooks.
- **Saves complete simulation results and synthesizes engineering reports**—including full 1D/2D/3D numerical field arrays (`outputs/results/simulation_fields.npz`), quantitative summary metrics (`outputs/data/summary_metrics.csv`), comparison figures (`outputs/plots/comparison.png`), and verified engineering reports (`outputs/reports/final_report.md`).
- **Audits physical laws and distills transferable experience**—verifying conservation, Discrete Maximum Principle bounds, and boundary satisfaction before recording causal edit attributions (`<edits>`) and reusable lessons in `library/<task_id>/EXPERIENCE.md`, `skills_snapshot/`, and `library/INDEX.md` behind a single atomic Human-in-the-Loop (HITL) gate.

---

## Theoretical Foundation: `Agent = Frozen Model + Editable Harness`

An LLM agent's real-world capability depends not only on its frozen model weights ($\mathcal{M}$), but also on its surrounding execution harness ($\mathcal{H}$)—the prompts, memory, skills, tools, scripts, and orchestration through which the model performs tasks ($\text{Agent} = \text{Model} + \text{Harness}$).

### File-Mounted `HARNESS²` Components (`harness/`)

| Runtime Group | Component ID & Path | Activation | Role in Scientific Simulation |
| :--- | :--- | :--- | :--- |
| **Context Construction**<br>*(What enters LLM context, and when)* | **`C1` System Prompt**<br>`harness/systemprompt.md` | **Every call** (Always-On) | Defines the core scientific computing persona and top-level 3-node closed-loop procedure followed across all tasks. |
| | **`C2` Guardrails**<br>`harness/guiderails.md` | **Every call** (Always-On) | Hard physical and operational constraints (read-only `tasks/`, non-overwriting `workspace/<task_id>/` rules, zero hardcoded constants in solvers, mandatory output verification, and verbatim skill triggers). |
| | **`C3` Long-Term Memory**<br>`harness/LongTermMEMORY.md` | **Every call** (Always-On) | Durable numerical computing facts persisted across tasks (headless `matplotlib.use('Agg')`, dynamic parameter loading from `specs/simulation_spec.json`, Discrete Maximum Principle bounds). |
| | **`C6` Skills**<br>`harness/skills/<name>/SKILL.md`<br>`harness/skills/<name>/scripts/*` | **On demand** (Skill loader) | Specialist 1D/2D/3D numerical procedures and companion checker scripts (`check_stencil.py`) loaded into context *only* when triggered, costing zero tokens when skipped or empty. |
| **Tool Interface**<br>*(What the model can call, and how)* | **`C4` Tool Guidance**<br>`harness/toolguidance.md` | **Every call** (Always-On) | Instructions on invoking MCP file I/O and debug inspection tools (`mcp_read_file`, `mcp_inspect_results`, `mcp_search_workspace`), external parameter loading, and CPU/GPU array idioms. |
| | **`C7` Scripts**<br>`harness/scripts/<name>.py` | **On demand** (Subprocess) | Deterministic Python capabilities (`hardware_probe.py`, `verify_outputs.py`, `inspect_results.py`, `solver_template.py`) executed deterministically. |
| **Middleware Control**<br>*(Outside model discretion)* | **`C8` Plugin**<br>`harness/plugins/runtime_middleware.py` | **Inside tool loop** (Every execution) | Deterministic hooks around subprocess execution that truncate oversized log lines, append OOM/headless hints, and fire deliverable watchdogs. |

### Phase-Sliced Externalized Node Prompt Library (`harness/nodes/*.md`)
Each of the 3 unified LangGraph nodes loads its phase-specific instructions from a single markdown file in `harness/nodes/` using `<!-- PHASE:<node_phase> -->` markers:
- **`harness/nodes/planner.md`**: Slices `requirement`, `pre_audit`, and `plan` phases.
- **`harness/nodes/solver_reporter.md`**: Slices `derive`, `audit_derivation`, `code`, `diagnose`, and `report` phases.
- **`harness/nodes/judge_reflector.md`**: Slices `post_audit` and `reflect` phases.
- **LaTeX-Safe Raw Mounting**: All `.md` files are mounted raw without calling `.format()` on them so LaTeX equations (such as `\frac{\partial^2 T}{\partial x^2}`) never collide with template braces.

---

## 3-Node Hierarchical Closed-Loop Control Architecture

### Control Flowchart

```text
========================================================================================================
                         DECOUPLED TASK INPUTS & HARNESS MOUNT (harness/)
  [C1] systemprompt.md   [C2] guiderails.md   [C3] LongTermMEMORY.md   [C4] toolguidance.md
  [Nodes] harness/nodes/{planner,solver_reporter,judge_reflector}.md (Phase-Sliced via `node_phase`)
  [C6] skills/<name>/SKILL.md (Read-Only / On-Demand; Skipped if Empty)
  [C7] scripts/{hardware_probe.py, verify_outputs.py, inspect_results.py, solver_template.py}
  [C8] plugins/runtime_middleware.py
  [Decoupled Task Stream] tasks/<task_id>/<task_id>.md + tasks/<task_id>/*.txt (Strictly Read-Only)
========================================================================================================
                                                  │
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. UNIFIED PLANNER NODE (agent/nodes/planner.py + Phase-Sliced harness/nodes/planner.md)             │
│    [Unifies Requirement Parsing + Pre-Audit Feasibility Gate + Knowledge Lookup + Planning]          │
│                                                                                                      │
│    Entry Router: IF no `simulation_spec` OR `planner_entry_phase == "requirement"` -> Step 1.1       │
│                  ELSE (`planner_entry_phase == "plan"`) --------------------------> Step 1.3         │
│                                                                                                      │
│    ├── Step 1.1: REQUIREMENT & VARIABLE EXTRACTION (`node_phase = "requirement"`)                    │
│    │     - Read `tasks/<task_id>/<task_id>.md` & `*.txt`; extract parameters into variables          │
│    │     - Run `C7: hardware_probe.py` (CPU cores, free RAM, GPU backend)                            │
│    │     - Formulate/revise 1D/2D/3D `SimulationSpec` (incorporating any OOM/escalation feedback)    │
│    ├── Step 1.2: PRE-AUDIT FEASIBILITY GATE (`node_phase = "pre_audit"`, `req_retry < max_req`)      │
│    │     - Verify 1D/2D/3D PDE well-posedness (Dirichlet/Neumann/Robin BCs & ICs)                    │
│    │     - Verify `estimate_grid_memory_mb() <= memory_safety_fraction * free_ram_mb`                │
│    │     ├── IF ILL-POSED / RAM EXCEEDED (`req_retry < max_req`):                                    │
│    │     │     └──> Increment `req_retry += 1` -> Refine Grid/Spec in [ Step 1.1 ]                   │
│    │     ├── ELSE IF `req_retry >= max_req` (Pre-Audit Retries Exhausted):                           │
│    │     │     └──> Deterministically clamp grid (`nx,ny,nz`) to fit RAM & write `simulation_spec`   │
│    │     └── IF FEASIBLE & WELL-POSED:                                                               │
│    │           └──> Write `workspace/<task_id>/specs/simulation_spec.json` (CoW versioned)           │
│    └── Step 1.3: KNOWLEDGE LOOKUP & STRATEGY (`node_phase = "plan"`)                                 │
│          - Strict Lookup Priority (Skipped cleanly at zero cost if empty or mismatched PDE):         │
│              1st: Check `harness/skills/*/SKILL.md` -> populate `active_skills`                      │
│              2nd: Check `library/INDEX.md` -> load matched `library/<task_id>/EXPERIENCE.md`         │
│                   and `library/<task_id>/skills_snapshot/*/SKILL.md` into `active_library_entries`   │
│          - Compact prior trajectory into `rolling_summary` (OOM guard)                               │
│          - Set `needs_derivation` (True/False) & `solver_entry_phase` ("derive" or "code")           │
└─────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                  │ (Validated `simulation_spec.json` & Priors Ready)
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. UNIFIED SOLVER & REPORTER NODE (agent/nodes/solver_reporter.py + harness/nodes/solver_reporter.md)│
│    [Unifies DERIVER + DERIVATION AUDIT + CODE & DEBUG + EXECUTION + DIAGNOSER + REPORTER]            │
│                                                                                                      │
│    Entry Router (Resets `attempt = 0`, `deriv_retry = 0`):                                           │
│                  ├── IF `solver_entry_phase == "derive"` -> Step 2.1 (Derive Phase)                  │
│                  ├── IF `solver_entry_phase == "code"`   -> Step 2.3 (Code & Debug Phase)            │
│                  └── IF `solver_entry_phase == "report"` -> Step 2.6 (Report Synthesis Phase)        │
│                                                                                                      │
│    ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│    │ Step 2.1: DERIVE PHASE (`node_phase = "derive"`)                                           │    │
│    │  - Injects loaded `SKILL.md` + `EXPERIENCE.md` priors (if any; skipped if empty)           │    │
│    │  - Reads `derivation_audit_feedback`, `diagnostic_feedback`, & `intermediate_results_digest│    │
│    │  - Derives 1D/2D/3D PDE stencils, variable-k fluxes, Dirichlet/Neumann/Robin BCs, & CFL    │    │
│    │  - Writes `workspace/<task_id>/derivations/derivation.md` (CoW versioned)                  │    │
│    └─────────────────────────────────────────────┬──────────────────────────────────────────────┘    │
│                                                  ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│    │ Step 2.2: DERIVATION AUDIT PHASE (`node_phase = "audit_derivation"`) [MATH AUDIT GATE]     │    │
│    │  - Runs `check_stencil.py` (if applicable) & audits `derivation.md` against `SimSpec` for: │    │
│    │    (1) Stencil conservation & variable-k flux continuity; (2) Dirichlet/Neumann/Robin BCs; │    │
│    │    (3) Diagonal dominance / CFL Fourier stability bound; (4) Spec variable alignment       │    │
│    │    ├── IF MATH / BC / STABILITY FLAW (`deriv_retry < max_deriv_retries`):                  │    │
│    │    │     └──> Increment `deriv_retry += 1` -> Loop to [ Step 2.1 ] with audit feedback     │    │
│    │    ├── ELSE IF `deriv_retry >= max_deriv_retries` (Audit Retries Exhausted):               │    │
│    │    │     └──> Attach `derivation_audit_warning` -> Proceed to [ Step 2.3: CODE & DEBUG ]   │    │
│    │    └── IF MATHEMATICALLY VERIFIED (`is_valid == True`):                                    │    │
│    │          └──> Proceed to [ Step 2.3: CODE & DEBUG PHASE ]                                  │    │
│    └─────────────────────────────────────────────┬──────────────────────────────────────────────┘    │
│                                                  ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│    │ Step 2.3: CODE & DEBUG PHASE (`node_phase = "code"`) [Log & Intermediate Result Debugging] │    │
│    │  - Injects loaded `SKILL.md` + `EXPERIENCE.md` priors + verified `derivation.md`           │    │
│    │  - Direct Read-Only Debug Access to Prior Attempt Artifacts (Preserved in `outputs/`):     │    │
│    │      1. Execution Logs: Full extracted `stderr_traceback` + `log_tail` + on-demand         │    │
│    │         line-window inspection of `outputs/run_logs/run_<ts>.log` via `mcp_read_file`      │    │
│    │      2. Intermediate Field Results: `outputs/results/*.npz` (or latest `.versions/*.npz`   │    │
│    │         if prior run crashed early) summarized via `C7: inspect_results.py` /              │    │
│    │         `mcp_inspect_results` (shapes, min/max, NaN/Inf counts, & BC slices `T[0], T[-1]`) │    │
│    │      3. Intermediate Metrics & Code History: `outputs/data/*.csv`, full `scripts/solver.py`│    │
│    │         (up to 600 lines), `.versions/` history, and `inner_loop_trace`                    │    │
│    │  - Anti-Dead-Loop Guards: Stepped retry temperature (`T + dT * attempt`) + SHA-256         │    │
│    │    identical-code stagnation detector                                                      │    │
│    │  - Writes variable-driven `workspace/<task_id>/scripts/solver.py` (Incremental per-case    │    │
│    │    saving to `.npz`/`.csv`; zero hardcoded constants; CoW versioned)                       │    │
│    └─────────────────────────────────────────────┬──────────────────────────────────────────────┘    │
│                                                  ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│    │ Step 2.4: EXECUTE & VERIFY PHASE [Autonomous Inside-Workspace Execution — No Execution HITL│    │
│    │  - Checks missing packages in shared `.venv/` outside workspace -> [HITL GATE 🛑 if needed]│    │
│    │  - Clean-Slate Move Guard: Moves (`shutil.move`) prior `outputs/{results,data,plots}/*`    │    │
│    │    into `.versions/` right before launch so stale files never cause false-positives        │    │
│    │  - Runs `solver.py` autonomously in `workspace/<task_id>/` (Configurable timeout)          │    │
│    │  - Runs `C7: verify_outputs.py` (cross-checking all spec scenarios & paths on disk),       │    │
│    │    `C7: inspect_results.py` (building `intermediate_results_digest`), & `C8: middleware`   │    │
│    │    (Leaves newly generated outputs in place for Step 2.5 Diagnose & Step 2.3 Debug)        │    │
│    └─────────────────────────────────────────────┬──────────────────────────────────────────────┘    │
│                                                  ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│    │ Step 2.5: DIAGNOSE PHASE (`node_phase = "diagnose"`)                                       │    │
│    │  - Appends attempt record (`code_sha256`, `exec_status`, `npz_digest`, `issues`) to trace  │    │
│    │  - Inspects return code, `stderr_traceback`, `intermediate_results_digest`, & verification:│    │
│    │    ├── Code / Syntax / Plot Bug (`attempt < max_inner`):                                   │    │
│    │    │     └──> Increment `attempt += 1` -> Loop to [ Step 2.3: CODE & DEBUG PHASE ]         │    │
│    │    ├── NaN / Stencil Instability (`attempt < max_inner`):                                  │    │
│    │    │     └──> Increment `attempt += 1`, reset `deriv_retry = 0` -> Loop to [ Step 2.1 ]    │    │
│    │    │          (Re-audited by [ Step 2.2: DERIVATION AUDIT ] before re-coding!)             │    │
│    │    ├── Retries Exhausted / Strategic Error:                                                │    │
│    │    │     ├── IF `outer_loop_count < max_outer_loops`:                                      │    │
│    │    │     │     └──> Increment `outer_loop_count += 1` -> Escalate to [ 1. PLANNER NODE ]   │    │
│    │    │     │          (`entry = "requirement"` if OOM/Grid issue else `entry = "plan"`)      │    │
│    │    │     └── ELSE (Circuit Breaker Tripped):                                               │    │
│    │    │           └──> Proceed to [ Step 2.6: REPORT SYNTHESIS ] (Diagnostic report)          │    │
│    │    └── Converged & Verified (`ok == True`) ─────────────> Proceed to [ Step 2.6: REPORT ]  │    │
│    └─────────────────────────────────────────────┬──────────────────────────────────────────────┘    │
│                                                  ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│    │ Step 2.6: REPORT SYNTHESIS PHASE (`node_phase = "report"`)                                 │    │
│    │  - Cross-checks `.npz` (`intermediate_results_digest`), `.csv`, `.png`, & `.py` vs Spec    │    │
│    │      └──> IF MISSING SPEC ARTIFACT (`attempt < max_inner`):                                │    │
│    │            └──> Increment `attempt += 1` -> Loop to [ Step 2.3: CODE & DEBUG PHASE ]       │    │
│    │  - Writes `workspace/<task_id>/outputs/reports/final_report.md` (CoW versioned) &          │    │
│    │    verifies `final_report.md` exists and is non-empty on disk                              │    │
│    └─────────────────────────────────────────────┬──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                  │ (All Deliverables Verified & `final_report.md` Saved)
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. UNIFIED JUDGE & REFLECTOR NODE (agent/nodes/judge_reflector.py + harness/nodes/judge_reflector.md)│
│    [Unifies Post-Audit Scientific Verification + HARNESS² & RRSI Continual Experience Extraction]    │
│                                                                                                      │
│    ├── Step 3.1: POST-AUDIT SCIENTIFIC VERIFICATION (`node_phase = "post_audit"`)                    │
│    │     - Audits saved `final_report.md`, `intermediate_results_digest` (`.npz` bounds/BC slices),  │
│    │       CSV metrics, & plots for physical laws (conservation of flux, Maximum Principle, BCs)     │
│    │     ├── IF REJECTED AND `outer_loop_count < max_outer_loops` (`outer_loop_count += 1`):         │
│    │     │     ├── Discretization / BC Flaw ──> [ 2. SOLVER & REPORTER ] (`entry = "derive"`)        │
│    │     │     ├── Code / Plot Mismatch ──────> [ 2. SOLVER & REPORTER ] (`entry = "code"`)          │
│    │     │     ├── Report Prose Mismatch ─────> [ 2. SOLVER & REPORTER ] (`entry = "report"`)        │
│    │     │     ├── Strategy / Missing Case ───> [ 1. UNIFIED PLANNER ]   (`entry = "plan"`)          │
│    │     │     └── Spec / Grid Resize Issue ──> [ 1. UNIFIED PLANNER ]   (`entry = "requirement"`)   │
│    │     ├── IF REJECTED AND `outer_loop_count >= max_outer_loops` (Circuit Breaker Tripped):        │
│    │     │     └──> Skip Step 3.2 (Prevent library poisoning) -> Exit to [ FINAL OUTPUT ]            │
│    │     └── IF APPROVED (`is_approved == True` AND `verification_report["ok"] == True`):            │
│    │           └──> Proceed to [ Step 3.2: CONTINUAL EXPERIENCE EXTRACTION ]                         │
│    └── Step 3.2: CONTINUAL EXPERIENCE EXTRACTION (`node_phase = "reflect"`, HARNESS² + RRSI)         │
│          - Reads `inner_loop_trace`; attributes `<edits>` (`helped|hurt|no_effect|confounded`)       │
│          - Extracts transferable lesson (`cue`, `lesson`, `evidence`, `confidence`)                  │
│          - Validates HARNESS² Rules 1–3 and RRSI character budget cap:                               │
│            ├── IF VALIDATION FAILS OR HITL DECLINED ──> Skip library write -> Exit to [FINAL OUTPUT] │
│            └── IF VALID & [SINGLE ATOMIC HITL GATE 🛑] APPROVED:                                     │
│                  └──> Saves `library/<task_id>/EXPERIENCE.md` + `skills_snapshot/` +                 │
│                       updates `library/INDEX.md` (Ready for Task t+1 in [ 1. UNIFIED PLANNER ])      │
└─────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                  │ (Report Verified on Disk & Library Updated)
                                                  ▼
========================================================================================================
  [ FINAL OUTPUT VERIFIED IN workspace/<task_id>/outputs/ ]:
    results/simulation_fields.npz + data/summary_metrics.csv + plots/comparison.png + reports/final_report.md
    + library/<task_id>/EXPERIENCE.md + library/INDEX.md
========================================================================================================
```

---

## Project Directory Structure

```text
AutoSciAgent/
├── README.md                                 # Project documentation and 3-node architecture guide
├── requirements.txt                          # Framework and solver dependencies
├── .env                                      # API key and runtime mode flags
├── main.py                                   # CLI entrypoint for sequential task stream execution
├── .venv/                                    # 🌍 SHARED PYTHON/GPU ENVIRONMENT (Outside workspace -> HITL on install)
├── configs/
│   └── config.json                           # System, model, memory, retry, and path settings
│
├── tasks/                                    # 📥 DECOUPLED INPUT TASKS (Outside workspace — Strictly Read-Only!)
│   └── <task_id>/                            # Sequential task folders (e.g., task_01, task_02, task_03)
│       ├── <task_id>.md                      # High-level 1D/2D/3D PDE specification & JSON parameter block
│       └── *.txt                             # Optional companion parameter text files
│
├── harness/                                  # 🧬 HARNESS MOUNT + 3 PHASE-SLICED NODE PROMPTS (.md/.py)
│   ├── systemprompt.md                       # C1: Always-on core scientific persona & workflow
│   ├── guiderails.md                         # C2: Always-on physical/safety invariants & skill triggers
│   ├── LongTermMEMORY.md                     # C3: Always-on durable numerical & platform conventions
│   ├── toolguidance.md                       # C4: Always-on MCP tool usage, debug inspection, & CPU/GPU rules
│   ├── nodes/                                # 📄 EXTERNALIZED 3-NODE PROMPTS (Phase-sliced via `node_phase`)
│   │   ├── planner.md                        # 1. Unified Planner prompt (requirement, pre_audit, plan)
│   │   ├── solver_reporter.md                # 2. Unified Solver & Reporter prompt (derive, audit_derivation, code, diagnose, report)
│   │   └── judge_reflector.md                # 3. Unified Judge & Reflector prompt (post_audit, reflect)
│   ├── skills/                               # C6: On-demand specialist 1D/2D/3D PDE skills (skipped if empty)
│   │   └── numerical_pde/
│   │       ├── SKILL.md                      # 1D (3-point), 2D (5-point), & 3D (7-point) PDE stencils
│   │       └── scripts/
│   │           └── check_stencil.py          # Verifies stencil conservation weights
│   ├── scripts/                              # C7: Deterministic verification, array inspection & solver scripts
│   │   ├── __init__.py
│   │   ├── hardware_probe.py                 # Detects CPU cores, free RAM, and GPU (CUDA/CuPy/PyTorch)
│   │   ├── verify_outputs.py                 # Spec-aware verification of .npz/.npy, CSV rows, and PNGs
│   │   ├── inspect_results.py                # Summarizes .npz shapes, min/max, NaNs, & boundary slices for debugging
│   │   └── solver_template.py                # Variable-driven 1D/2D/3D solver reading specs/simulation_spec.json
│   └── plugins/                              # C8: Middleware hooks inside the execution/tool loop
│       ├── __init__.py
│       └── runtime_middleware.py             # Truncates noisy stdout, appends OOM hints, & runs watchdogs
│
├── library/                                  # 🧠 CONTINUAL IMPROVEMENT LIBRARY (Skipped if empty; Atomic HITL on write)
│   ├── INDEX.md                              # Maps physical/numerical cues to past EXPERIENCE.md entries
│   └── <task_id>/
│       ├── EXPERIENCE.md                     # Structured <edits> verdicts + transferable <experience>
│       └── skills_snapshot/                  # Adopted skills from completed tasks
│
├── core/                                     # 🛡️ CORE INFRASTRUCTURE, MCP DEBUG TOOLS & SECURITY JAIL
│   ├── __init__.py                           # Loads .env, config.json, and initializes Gemini LLM (or Mock)
│   ├── logger.py                             # Mode-aware logger (`logs/system/run_<timestamp>.log`)
│   ├── harness_manager.py                    # Mounts phase-sliced .md prompts, loads skills/EXPERIENCE.md, validates Rules 1-3
│   ├── hardware.py                           # Runs hardware discovery, estimates 1D/2D/3D RAM, & clamps oversized grids
│   ├── security.py                           # Enforces per-task `workspace/<task_id>/` jail & read-only `tasks/`
│   ├── tools_hitl.py                         # HITL gate for outside-workspace library writes & package installs
│   ├── tools_io.py                           # MCP I/O: CoW write, clean-slate output move guard, OOM-safe read
│   ├── tools_search.py                       # MCP debug tools: regex workspace grep & `mcp_inspect_results`
│   └── document_parser.py                    # Parses .md/.txt files & extracts parameter blocks into variables
│
├── agent/                                    # ⚙️ 3-NODE LANGGRAPH CLOSED-LOOP CONTROL SYSTEM
│   ├── __init__.py
│   ├── state.py                              # AutoSciState TypedDict & Pydantic structured output schemas
│   ├── graph.py                              # Compiles the 3-node StateGraph (`planner`, `solver_reporter`, `judge_reflector`)
│   └── nodes/
│       ├── __init__.py
│       ├── planner.py                        # 1. Unified Planner Node
│       ├── solver_reporter.py                # 2. Unified Solver & Reporter Node
│       └── judge_reflector.py                # 3. Unified Judge & Reflector Node
│
├── workspace/                                # 🔒 ISOLATED PER-TASK NON-OVERWRITING SANDBOXES
│   └── <task_id>/
│       ├── .versions/                        # Immutable Copy-on-Write archive (`manifest.json` + `.v001` files)
│       ├── .trash/                           # Soft-deleted workspace files
│       ├── specs/                            # `simulation_spec.json` (variables read by `scripts/solver.py`)
│       ├── derivations/                      # `derivation.md`
│       ├── scripts/                          # `solver.py`
│       └── outputs/
│           ├── run_logs/                     # `run_<timestamp>.log`
│           ├── results/                      # `simulation_fields.npz`
│           ├── data/                         # `summary_metrics.csv`
│           ├── plots/                        # `comparison.png`
│           └── reports/                      # `final_report.md`
│
├── logs/
│   └── system/

```

---

## Representative 1D, 2D & 3D PDE Examples (`tasks/<task_id>/<task_id>.md`)

All parameters (domain lengths, grid resolutions, time steps, boundary temperatures, thermal conductivities, and scenario sweeps) are stored in `tasks/<task_id>/<task_id>.md` (or `.txt` files) and loaded into variables at runtime.

**Example 1 (`tasks/task_0001/task_0001.md`) — 1D Transient Heat Conduction Equation**

   $$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}, \alpha = \frac{k_x}{\rho c_p}$$
   
   Discretized via the 3-point FTCS scheme with automatic Courant–Friedrichs–Lewy Fourier stability sub-stepping
   
   $$\text{Fo} = \frac{\alpha \Delta t}{(\Delta x)^2} \le \text{Fo}_{\max}$$

**Example 2 (`tasks/task_0002/task_0002.md`) — 2D Transient & Steady-State Orthotropic Heat Conduction**

   $$k_x \frac{\partial^2 T}{\partial x^2} + k_y \frac{\partial^2 T}{\partial y^2} = 0$$
   
   Discretized via the 5-point orthotropic Finite Difference stencil across `Case_A_Baseline_Isotropic`, `Case_B_Elevated_Boundary`, and `Case_C_Anisotropic_Conductivity`.

**Example 3 (`tasks/task_0003/task_0003.md`) — 3D Steady-State Anisotropic Heat Conduction**

   $$\frac{\partial}{\partial x} \left( k_{x} \frac{\partial T}{\partial x} \right) + \frac{\partial}{\partial y} \left( k_{y} \frac{\partial T}{\partial y} \right) + \frac{\partial}{\partial z} \left( k_{z} \frac{\partial T}{\partial z} \right) = 0$$
   
   with Dirichlet bottom contact $T(x,y,0) = T_{\text{bed}}(x,y)$, Robin top convection 
   
   $$-k_{\text{top}} \frac{\partial T}{\partial z}\big|_{z=L_z} = h(T_{\text{top}} - T_\infty)$$
   
   and adiabatic Neumann lateral walls across **Uniform**, **Linear**, and **Random Stratified** through-thickness conductivity distributions $k_{z,i,j}(z)$.

---

## Installation & Quick Start

```powershell
# Create and activate the shared virtual environment (.venv/)
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\activate
pip install -r requirements.txt

# Run all tasks in tasks/ sequentially
python main.py
```
