# AutoSciAgent

**AutoSciAgent** is an autonomous, general-purpose **Physical Simulation and Partial Differential Equation (PDE) Solver Agent**. 
We aim at developing a secure, self-verifying scientific computing workflow, based on the following concepts: **Control Theory** and **Agent = Model + Harness** ([HARNESS²](https://arxiv.org/abs/2610.14224) and [RRSI](https://arxiv.org/abs/2609.24972)).

Given a high-level physical simulation task (`workspace/inputs/task.md`), markdown harness instructions (`harness/*.md`), and optional reference documents (`.pdf`, `.docx`, `.md`) in `workspace/inputs/references/`, the agent autonomously:

* Parses requirements and digests scientific literature using isolated subagent contexts.

* Evaluates PDE mathematical well-posedness and CPU/GPU memory feasibility before execution.

* Curates Judge-approved domain equations and material properties into a SHA-256 deduplicated **ChromaDB RAG** store (`data/vector_store/`).

* Derives numerical discretizations (Finite Difference, Finite Volume, or Finite Element stencils, boundary conditions, and stability criteria) when required.

* Writes standalone, CPU/GPU-adaptive Python solvers (`cupy`/`torch` on CUDA GPUs; `numpy`/`scipy` on CPU) and multi-scenario comparison visualizers from scratch.

* Executes solvers inside an isolated virtual environment (`workspace/.venv`) with hard timeouts, deterministic output verification (`verify_outputs.py`), and runtime middleware hooks.

* Diagnoses runtime tracebacks, numerical divergence (`NaN`/`Inf`), or missing deliverables in a closed feedback loop.

* Synthesizes quantitative engineering reports (`final_report.md`) and extracts causal experience records (`EXPERIENCE.md`) and reusable skills (`SKILL.md`) for future tasks.

## Theoretical Foundation: `Agent = Frozen Model + Editable Harness`

An LLM agent's real-world capability depends not only on its frozen model weights ($\mathcal{M}$), but also on its surrounding execution harness ($\mathcal{H}$)—the prompts, memory, skills, tools, scripts, and orchestration through which the model performs tasks ($\text{Agent} = \text{Model} + \text{Harness}$).

### The 8 File-Mounted `HARNESS²` Components (`harness/`)
Following the `HARNESS²` runtime decomposition, `AutoSciAgent` exposes $8$ editable component groups mounted as plain files on disk so that every behavioral modification is a transparent file diff:

| Runtime Group | Component ID & Path | Activation | Role in Scientific Simulation & Cost Profile |
| :--- | :--- | :--- | :--- |
| **Context Construction**<br>*(What enters LLM context, and when)* | **`C1` System Prompt**<br>`harness/systemprompt.md` | **Every call** (Always-On) | Defines the core scientific computing persona and top-level closed-loop procedure followed across all tasks. *High token cost: competes with the agent's output budget on every turn*. |
| | **`C2` Guardrails**<br>`harness/guiderails.md` | **Every call** (Always-On) | Hard physical and operational constraints that hold regardless of task or tool (read-only inputs, non-overwriting workspace rules, mandatory output verification, and verbatim skill trigger lines). |
| | **`C3` Long-Term Memory**<br>`harness/LongTermMEMORY.md` | **Every call** (Always-On) | Durable numerical computing facts persisted across tasks in the domain (e.g., headless `matplotlib.use('Agg')`, relative workspace paths, Discrete Maximum Principle bounds). |
| | **`C6` Skills**<br>`harness/skills/<name>/SKILL.md`<br>`harness/skills/<name>/scripts/*` | **On demand** (Skill loader) | Specialist numerical procedures (e.g., `fdm_2d_heat`) and companion checker scripts loaded into context *only* when triggered, costing zero tokens on tasks that do not need them. |
| **Tool Interface**<br>*(What the model can call, and how)* | **`C4` Tool Guidance**<br>`harness/toolguidance.md` | **Every call** (Always-On) | Instructions on when and how to invoke MCP file I/O tools, bounded line-window reads (`max_lines=150`), and CPU/GPU array idioms. |
| | **`C7` Scripts**<br>`harness/scripts/<name>.py` | **On demand** (Subprocess) | Deterministic Python capabilities (`hardware_probe.py`, `verify_outputs.py`) executed via shell rather than reasoned through token-by-token. |
| **Orchestration**<br>*(How contexts are split)* | **`C5` Subagents**<br>`harness/sub_agents/<name>/AGENT.md` | **On demand** (Subagent call) | Delegated worker (`reference_digester`) running in its own isolated context window and returning only a compact summary, preventing large PDFs from overflowing the main context. |
| **Middleware Control**<br>*(Outside model discretion)* | **`C8` Plugin**<br>`harness/plugins/runtime_middleware.py` | **Inside tool loop** (Every execution) | Deterministic hooks around subprocess execution that enforce behavior rather than instruct it: truncating oversized log lines, appending targeted OOM/headless hints, and firing deliverable watchdogs. |

### Externalized Node Prompt Library (`harness/nodes/*.md`)
In addition to `C1–C8`, every LangGraph node loads its role-specific instructions from a dedicated markdown file in `harness/nodes/` (`requirement.md`, `planner.md`, `deriver.md`, `coder.md`, `diagnoser.md`, `reporter.md`, `judge.md`, `reflector.md`).
* **Zero Code Edits for Prompt Tuning**: You can refine how any node plans, derives math, writes code, or audits physics simply by editing its `.md` file in VS Code without touching a single `.py` file.
* **LaTeX-Safe Raw Loading**: Because scientific prompts contain LaTeX braces (such as `\frac{\partial^2 T}{\partial x^2}`) and template engines with strict undefined checks crash on unescaped `{{` or `{%` tags, `core/harness_manager.py` mounts all `.md` files raw without calling `.format()` on them, appending dynamic runtime state in a separate structured section.

### `HARNESS²` Validation Rules & `RRSI` Regularization
Before any harness tree or self-improvement update is accepted, `core/harness_manager.py` enforces four strict invariants:
* **Rule 1 — Verbatim Skill Trigger Check**: Every skill directory `harness/skills/<name>/SKILL.md` must have its exact `<name>` appear verbatim in `systemprompt.md` or `guiderails.md` (e.g., *"When solving 2D steady or transient heat conduction or diffusion PDEs, load the `fdm_2d_heat` skill."*).
* **Rule 2 — No Unresolved Template Braces**: Always-on markdown files (`C1–C4`) are scanned to ensure no raw Jinja `{{` or `{%` tags are present.
* **Rule 3 — Referenced Script Existence & Syntax Check**: Every `scripts/<file>.py` referenced in always-on text or skills must exist on disk and pass Python bytecode compilation (`py_compile`).
* **RRSI Token-Budget Regularization**: Because always-on prose (`C1–C4`) is paid for on every call and competes with the model's output token budget, the combined length of `systemprompt.md`, `guiderails.md`, `LongTermMEMORY.md`, and `toolguidance.md` is capped at `24,000` characters (`rrsi_max_always_on_chars`). Newly learned domain procedures must be encapsulated as on-demand `C6` skills (`harness/skills/<name>/SKILL.md`) or stored in `library/`.

---

## Closed-Loop Control System Architecture

`AutoSciAgent` models scientific simulation as a multi-loop feedback control system with bounded retry counters (`req_retry_count`, `inner_retry_count`, `outer_loop_count`) to guarantee Bounded-Input Bounded-Output (BIBO) stability.

### Control Flowchart

```text
========================================================================================================
                         INPUTS & 8-COMPONENT HARNESS MOUNT (harness/)
  [C1] systemprompt.md   [C2] guiderails.md   [C3] LongTermMEMORY.md   [C4] toolguidance.md
  [Nodes] harness/nodes/{requirement,planner,deriver,coder,diagnoser,reporter,judge,reflector}.md
  [C5] sub_agents/reference_digester/AGENT.md          [C6] skills/fdm_2d_heat/SKILL.md
  [C7] scripts/{hardware_probe.py, verify_outputs.py}  [C8] plugins/runtime_middleware.py
  [Task] workspace/inputs/task.md + workspace/inputs/references/*.pdf | *.docx | *.md
========================================================================================================
                                                  │
                                                  ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ 1. REQUIREMENT NODE (agent/nodes/requirement.py)          │
                    │  - Loads `harness/nodes/requirement.md` + C1-C4           │
                    │  - Runs `C7: hardware_probe.py` (CPU cores, RAM, GPU)     │
                    │  - Digests `.pdf/.doc/.md` via `C5: reference_digester`   │
                    │  - Writes `workspace/specs/simulation_spec.json` (CoW)    │
                    │  - Sets `judge_stage = "pre_audit"`                       │
                    └─────────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ 2A. JUDGE NODE [PRE-AUDIT] (agent/nodes/judge.py)         │
                    │  - Loads `harness/nodes/judge.md` (Stage: pre_audit)      │
                    │  - Verifies PDE well-posedness (BCs/ICs) & RAM/VRAM limits│
                    │      └──> IF ILL-POSED / INFEASIBLE ──> [ 1. REQUIREMENT ]│
                    │  - Reviews digested reference summaries for RAG value:    │
                    │      └──> IF STORE_IN_RAG = True:                         │
                    │           [HITL GATE 🛑 (Outside Workspace)]              │
                    │           ──> Indexes into `data/vector_store/` (SHA-dedup│
                    └─────────────────────────────┬─────────────────────────────┘
                                                  │ (Feasible & Spec Approved)
                                                  ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ 3. PLANNER NODE (agent/nodes/planner.py)                  │
                    │  - Loads `harness/nodes/planner.md` + `library/INDEX.md`  │
                    │  - Compacts past trajectory into `rolling_summary` (OOM)  │
                    │  - Activates matching `C6: skills/<name>/SKILL.md`        │
                    │  - Evaluates: `needs_derivation` (True / False)           │
                    └──────────────┬──────────────────────────────┬─────────────┘
         (needs_derivation = True) │                              │ (needs_derivation = False)
                                   ▼                              │
         ┌──────────────────────────────────────────────────┐     │
         │ 4. DERIVER NODE (agent/nodes/deriver.py)         │     │
         │  - Loads `harness/nodes/deriver.md` + RAG priors │     │
         │  - Derives discretized PDE stencils (FDM/FVM),   │     │
         │    anisotropic weights, BCs, & CFL stability     │     │
         │  - Writes `workspace/derivations/derivation.md`  │     │
         │    (CoW archives prior versions to `.versions/`) │     │
         └─────────────────────────┬────────────────────────┘     │
                                   └──────────────┬───────────────┘
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               INNER ACTUATION, PLANT & DIAGNOSTIC LOOP                               │
│                                                                                                      │
│                   ┌───────────────────────────────────────────────────────────┐                      │
│                   │ 5. CODER NODE (agent/nodes/coder.py)                      │                      │
│                   │  - Loads `harness/nodes/coder.md` + active `SKILL.md`     │                      │
│                   │  - Reads `derivation.md` (if present) + Hardware Profile  │                      │
│                   │  - Writes CPU/GPU-adaptive solver & sweep plotter to      │                      │
│                   │    `workspace/scripts/solver_2d.py` (CoW Non-Overwriting) │                      │
│                   └─────────────────────────────┬─────────────────────────────┘                      │
│                                                 ▼                                                    │
│                   ┌───────────────────────────────────────────────────────────┐                      │
│                   │ 6. EXECUTION NODE (agent/nodes/execution.py)              │                      │
│                   │  - Archives prior `outputs/{data,plots}` to `.versions/`  │                      │
│                   │  - Runs `solver_2d.py` in `workspace/.venv` (60s Timeout) │                      │
│                   │  - Runs `C7: verify_outputs.py` (checks CSV NaNs & PNGs)  │                      │
│                   │  - Applies `C8: runtime_middleware.py` (truncates noise)  │                      │
│                   │  - Saves full log to `outputs/run_logs/run_<N>.log`;      │                      │
│                   │    stores only file pointer + 40-line tail in state       │                      │
│                   └─────────────────────────────┬─────────────────────────────┘                      │
│                                                 ▼                                                    │
│                   ┌───────────────────────────────────────────────────────────┐                      │
│                   │ 7. DIAGNOSER NODE (agent/nodes/diagnoser.py)              │                      │
│                   │  - Loads `harness/nodes/diagnoser.md`                     │                      │
│                   │  - Inspects log tail + `[VERIFICATION_REPORT]`:           │                      │
│                   │    ├── Syntax / API / GPU-Fallback Bug ──> [ 5. CODER ]   │                      │
│                   │    ├── NaN Divergence / Stencil Error ───> [ 4. DERIVER ] │                      │
│                   │    ├── OOM / Circuit Breaker (retries=3) > [ 3. PLANNER ] │                      │
│                   │    └── Converged & Verified ─────────────> [ 8. REPORTER ]│                      │
│                   └─────────────────────────────┬─────────────────────────────┘                      │
└─────────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                                  │ (All Deliverables Verified on Disk)
                                                  ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ 8. REPORTER NODE (agent/nodes/reporter.py)                │
                    │  - Loads `harness/nodes/reporter.md`                      │
                    │  - Cross-checks `.png`, `.csv`, `.py` against `SimSpec`   │
                    │      └──> IF MISSING SWEEP / ARTIFACT ──> [ 3. PLANNER ]  │
                    │  - Writes `workspace/outputs/reports/final_report.md`     │
                    │  - Sets `judge_stage = "post_audit"`                      │
                    └─────────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ 2B. JUDGE NODE [POST-AUDIT] (agent/nodes/judge.py)        │
                    │  - Loads `harness/nodes/judge.md` (Stage: post_audit)     │
                    │  - Audits physical conservation, Maximum Principle, BCs,  │
                    │    and quantitative consistency between CSVs, plots & text│
                    │      ├── Math / Discretization Flaw ────> [ 4. DERIVER ]  │
                    │      ├── Code / Visualization Mismatch ─> [ 5. CODER ]    │
                    │      └── Missing Comparison Scenario ───> [ 3. PLANNER ]  │
                    └─────────────────────────────┬─────────────────────────────┘
                                                  │ (Scientific Audit Approved)
                                                  ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │ 9. REFLECTOR NODE (agent/nodes/reflector.py)              │
                    │  - Loads `harness/nodes/reflector.md` (HARNESS² + RRSI)   │
                    │  - Attributes `<edits>` (`helped|hurt|no_effect|confound`)│
                    │  - Extracts reusable `SKILL.md` + `EXPERIENCE.md`         │
                    │  - Validates HARNESS² Rules 1-3 & RRSI 24k-char cap       │
                    │  - [HITL GATE 🛑 (Outside Workspace)] -> Updates `library`│
                    └─────────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
========================================================================================================
    [ FINAL OUTPUT ]: workspace/outputs/reports/final_report.md + solver_2d.py + .png plots + .csv
========================================================================================================
```

### Node-by-Node Responsibilities

* **`requirement.py` (Reference Signal Generator & Literature Digester)**:
   Reads `workspace/inputs/task.md`, probes hardware capabilities via `hardware_probe.py`, and digests any `.pdf`, `.docx`, `.doc`, `.md`, or `.txt` files in `workspace/inputs/references/` using the isolated `reference_digester` subagent (`C5`). Emits a structured `SimulationSpec` (`workspace/specs/simulation_spec.json`) and sets `judge_stage = "pre_audit"`.
* **`judge.py` — Stage 1 (`pre_audit`: Feasibility & RAG Gate)**:
   Evaluates whether the PDE and boundary conditions in `SimulationSpec` are mathematically well-posed and whether the estimated grid memory (`estimate_grid_memory_mb`) fits safely within available system RAM or GPU VRAM. Reviews `requirement`'s digested reference summaries and decides which documents merit permanent indexing into ChromaDB (`data/vector_store/`).
* **`planner.py` (Master Controller & Long-Horizon Memory Manager)**:
   Reads `library/INDEX.md` and `SimulationSpec`, compresses prior retry history into a bounded `rolling_summary` string to prevent context OOM, selects relevant on-demand skills (`C6`), and determines whether mathematical derivation is required (`needs_derivation = True` routes to `deriver`; `False` routes directly to `coder`).
* **`deriver.py` (Conditional Mathematical Discretizer)**:
   Queries the RAG store for governing physics priors and derives the exact numerical scheme (e.g., 5-point orthotropic finite-difference stencils, matrix coefficients, Neumann/Dirichlet boundary formulas, and Courant–Friedrichs–Lewy stability bounds), saving the result to `workspace/derivations/derivation.md`.
* **`coder.py` (Solver & Visualization Actuator)**:
   Combines `SimulationSpec`, the hardware profile, active `SKILL.md` files, and `derivation.md` (if generated) to write a self-contained, CPU/GPU-adaptive Python solver and Matplotlib comparison plotter to `workspace/scripts/solver_2d.py`.
* **`execution.py` (Sandboxed Plant, Verifier & Middleware Runner)**:
   Archives any existing output files via Copy-on-Write, executes `solver_2d.py` inside `workspace/.venv` under a 60-second timeout, runs `harness/scripts/verify_outputs.py` (`C7`) to reopen and validate all generated `.csv` and `.png`, and applies `runtime_middleware.py` (`C8`). Full output is streamed to `workspace/outputs/run_logs/run_<timestamp>.log`, while only the file pointer and a 40-line tail are stored in `AutoSciState`.
* **`diagnoser.py` (Inner-Loop Feedback Sensor)**:
   Inspects the execution status, log tail, CSV previews, and `[VERIFICATION_REPORT]`. Routes Python syntax/import/device bugs back to `coder`, numerical divergence (`NaN`/`Inf`) or stencil errors back to `deriver`, grid OOM or exhausted inner retries (`max_inner_retries = 3`) back to `planner`, and verified runs forward to `reporter`.
* **`reporter.py` (Deliverable Synthesizer & Requirement Cross-Checker)**:
   Cross-checks generated plots (`outputs/plots/*.png`) and data tables (`outputs/data/*.csv`) against `requirement`'s `SimulationSpec`. If deliverables are missing, routes back to `planner`; otherwise synthesizes `workspace/outputs/reports/final_report.md` and sets `judge_stage = "post_audit"`.
* **`judge.py` — Stage 2 (`post_audit`: Final Scientific & Consistency Auditor)**:
   Audits the final report, CSV metrics, derivations, and plots for physical laws (Discrete Maximum Principle, energy conservation, directional heat flux trends) and internal consistency. Routes discrepancies back to `deriver`, `coder`, or `planner` (up to `max_outer_loops = 2`), or approves the package and routes to `reflector`.
* **`reflector.py` (`HARNESS²` + `RRSI` Continual Experience Extractor)**:
    Audits the complete task record, assigns causal verdicts (`helped`, `hurt`, `no observed effect`, or `confounded`) to components, verifies `HARNESS²` Rules 1–3 and the `RRSI` character cap, and writes transferable lessons to `library/<task_slug>/EXPERIENCE.md` and `library/INDEX.md`.

---

## Project Directory Structure

```text
AutoSciAgent/
├── README.md                                 # Project documentation and architecture guide
├── requirements.txt                          # Framework and testing dependencies
├── .env                                      # API key and runtime mode flags
├── main.py                                   # CLI entrypoint for running the closed-loop agent
├── configs/
│   └── config.json                           # Model settings, loop limits, timeouts, and paths
│
├── harness/                                  # 🧬 8-COMPONENT HARNESS + NODE PROMPTS (.md/.py)
│   ├── systemprompt.md                       # C1: Always-on core scientific persona & workflow
│   ├── guiderails.md                         # C2: Always-on physical/safety invariants & skill triggers
│   ├── LongTermMEMORY.md                     # C3: Always-on durable numerical & platform conventions
│   ├── toolguidance.md                       # C4: Always-on MCP tool usage, OOM slicing, & CPU/GPU rules
│   ├── nodes/                                # 📄 EXTERNALIZED NODE PROMPTS (Modify .md, never .py!)
│   │   ├── requirement.md                    # Prompt template for requirement.py
│   │   ├── planner.md                        # Prompt template for planner.py
│   │   ├── deriver.md                        # Prompt template for deriver.py
│   │   ├── coder.md                          # Prompt template for coder.py
│   │   ├── diagnoser.md                      # Prompt template for diagnoser.py
│   │   ├── reporter.md                       # Prompt template for reporter.py
│   │   ├── judge.md                          # Prompt template for judge.py (pre_audit & post_audit)
│   │   └── reflector.md                      # Prompt template for reflector.py (HARNESS² + RRSI)
│   ├── sub_agents/                           # C5: Isolated-context subagent markdown definitions
│   │   └── reference_digester/
│   │       └── AGENT.md                      # Digests lengthy PDFs/docs in isolated context (prevents OOM)
│   ├── skills/                               # C6: On-demand specialist PDE skills (zero cost until triggered)
│   │   └── fdm_2d_heat/
│   │       ├── SKILL.md                      # 2D steady/transient isotropic & orthotropic FDM stencils
│   │       └── scripts/
│   │           └── check_stencil.py          # Verifies stencil weights & discrete Maximum Principle
│   ├── scripts/                              # C7: Deterministic verification & hardware probe scripts
│   │   ├── __init__.py
│   │   ├── hardware_probe.py                 # Detects CPU cores, free RAM, and GPU (CUDA/CuPy/PyTorch)
│   │   └── verify_outputs.py                 # Reopens & verifies CSVs (no NaN/Inf) and PNG files post-run
│   └── plugins/                              # C8: Middleware hooks inside the execution/tool loop
│       ├── __init__.py
│       └── runtime_middleware.py             # Truncates noisy stdout, appends OOM hints, & runs watchdogs
│
├── library/                                  # 🧠 HARNESSING-LEVEL RECURSION LIBRARY (Outside workspace -> HITL)
│   ├── INDEX.md                              # Maps physical/numerical cues to past EXPERIENCE.md entries
│   └── seed_2d_elliptic/
│       ├── EXPERIENCE.md                     # Structured <edits> verdicts + transferable <experience>
│       └── skills_snapshot/                  # Adopted skills from completed tasks
│
├── core/                                     # 🛡️ CORE INFRASTRUCTURE, MCP TOOLS & SECURITY JAIL
│   ├── __init__.py                           # Loads .env, config.json, and initializes Gemini LLM (or Mock)
│   ├── logger.py                             # Mode-aware logger (`logs/system/run_<timestamp>.log`)
│   ├── harness_manager.py                    # Mounts .md prompts, validates Rules 1-3, enforces RRSI caps
│   ├── hardware.py                           # Runs hardware discovery & estimates grid memory footprint
│   ├── security.py                           # Enforces non-overwriting workspace rules & outside-workspace locks
│   ├── tools_hitl.py                         # HITL gate for outside-workspace edits, RAG writes, & execution
│   ├── tools_io.py                           # MCP I/O: Copy-on-Write versioned write, OOM-safe read, soft-delete
│   ├── tools_search.py                       # MCP search tools: workspace file search, regex grep, & RAG query
│   ├── document_parser.py                    # Extracts text from .pdf, .docx, .doc, .md, .txt
│   └── rag_engine.py                         # SHA-256 deduplicated ChromaDB vector store
│
├── agent/                                    # ⚙️ LANGGRAPH CLOSED-LOOP CONTROL SYSTEM
│   ├── __init__.py
│   ├── state.py                              # AutoSciState TypedDict & Pydantic structured output schemas
│   ├── graph.py                              # Compiles the 9-node StateGraph & conditional routing edges
│   └── nodes/
│       ├── __init__.py
│       ├── requirement.py                    # Reads task.md + harness .md + digests references via subagent
│       ├── judge.py                          # Dual-stage: (1) Feasibility & RAG gate; (2) Final physics audit
│       ├── planner.py                        # Master controller, rolling_summary OOM guard, deriver router
│       ├── deriver.py                        # Derives PDE discretizations, BCs, & stability bounds
│       ├── coder.py                          # Writes CPU/GPU-adaptive Python solver & sweep visualizations
│       ├── execution.py                      # Runs solver in workspace/.venv + C7 verify + C8 middleware
│       ├── diagnoser.py                      # Inner-loop error sensor routing to coder / deriver / planner
│       ├── reporter.py                       # Audits deliverables against requirement & writes final_report.md
│       └── reflector.py                      # Extracts reusable skills & causal EXPERIENCE.md (RRSI-guarded)
│
├── workspace/                                # 🔒 COLLABORATIVE NON-OVERWRITING SANDBOX
│   ├── .venv/                                # Isolated Python environment for executing generated PDE solvers
│   ├── .versions/                            # Immutable Copy-on-Write archive (`manifest.json` + `.v001` files)
│   ├── .trash/                               # Soft-deleted workspace files (prevents permanent data loss)
│   ├── inputs/                               # Strictly read-only task inputs
│   │   ├── task.md                           # Test example: 2D Heat Simulation with BC & Property Sweeps
│   │   └── references/                       # Reference documents (.pdf, .docx, .md)
│   │       └── 2d_heat_conduction_reference.pdf
│   ├── specs/                                # Saved SimulationSpec (`simulation_spec.json`)
│   ├── derivations/                          # Mathematical derivations (`derivation.md`) from deriver.py
│   ├── scripts/                              # Generated PDE solvers (`solver_2d.py`) from coder.py
│   └── outputs/
│       ├── run_logs/                         # Full subprocess stdout/stderr logs (`run_<timestamp>.log`)
│       ├── data/                             # Quantitative comparison metrics (`summary_metrics.csv`)
│       ├── plots/                            # Multi-panel heatmaps & centerline comparison plots (`.png`)
│       └── reports/                          # Final engineering report (`final_report.md`)
│
├── data/
│   └── vector_store/                         # Persistent ChromaDB storage (Outside workspace -> Judge + HITL)
├── logs/
│   └── system/                               # Unique per-run agent logs (`run_<YYYYMMDD_HHMMSS>.log`)
└── tests/                                    # 🧪 AUTOMATED UNIT & WORKFLOW TEST SUITE
    ├── __init__.py
    ├── test_nodes.py                         # Unit tests for all 9 nodes, CoW non-overwrite I/O, & validator
    └── test_workflow.py                      # End-to-end LangGraph closed-loop workflow & routing tests
```

---

## Safety, Non-Overwriting Workspace & Resource Management

### Copy-on-Write (CoW) Non-Overwriting `workspace/`
All nodes can read, write, revise, add, and remove files inside `workspace/`, but **destructive in-place overwriting is physically blocked**:
* **Automatic Version Archiving**: Whenever any node writes to a file path that already exists in `workspace/` (e.g., `coder.py` revising `workspace/scripts/solver_2d.py` or `deriver.py` revising `workspace/derivations/derivation.md`), `smart_write_file()` in `core/tools_io.py` automatically copies the existing file to `workspace/.versions/<subdir>/<stem>.v<NNN>_<timestamp><suffix>` and records its SHA-256 hash, timestamp, and author node in `workspace/.versions/manifest.json` before writing the new content.
* **Pre-Execution Output Snapshots**: Before `execution.py` runs a solver script, any existing files in `workspace/outputs/data/` and `workspace/outputs/plots/` are archived via Copy-on-Write so prior simulation results are never lost.
* **Soft Deletion (`workspace/.trash/`)**: Calling `smart_remove_item()` inside `workspace/` moves the file or folder to `workspace/.trash/<timestamp>_<name>` rather than permanently unlinking it.
* **Read-Only Inputs**: `core/security.py` blocks any write or delete operation targeting `workspace/inputs/` or `workspace/.venv/`.

### Human-in-the-Loop (HITL) Gate Outside `workspace/`
`core/security.py` classifies every resolved path as either `INSIDE_WORKSPACE` or `OUTSIDE_WORKSPACE` (`harness/`, `library/`, `data/vector_store/`). Any operation that:
* Adds, modifies, or removes files/folders outside `workspace/`
* Indexes new documents into the persistent RAG store (`data/vector_store/`)
* Installs extra Python packages into `workspace/.venv`
* Executes generated solver scripts

pauses and prompts the human operator in the terminal (`Approve this operation? [Y/n]:`) via `core/tools_hitl.py`.

### Four-Tier Long-Horizon OOM Protection
* **Subagent Reference Digestion (`C5`)**: Large `.pdf` and `.docx` files in `workspace/inputs/references/` are processed in an isolated subagent call (`reference_digester`) that returns only extracted equations and property tables under 400 words.
* **Pre-Execution Grid Memory Estimation**: `core/hardware.py` computes the memory footprint of the proposed PDE mesh (`nx * ny * nz * 8 bytes * num_arrays`) during `judge` (`pre_audit`) and rejects meshes that exceed 80% of free RAM/VRAM.
* **Streamed Subprocess Logs & Middleware Truncation (`C8`)**: Subprocess `stdout`/`stderr` is written directly to `workspace/outputs/run_logs/run_<timestamp>.log`. Only the file path pointer and the last 40 lines (`max_log_tail_lines = 40`) enter `AutoSciState`, with lines over 300 characters truncated by `runtime_middleware.py`.
* **Bounded File Reads & Rolling Summary Compaction**: `smart_read_file()` reads bounded windows (`max_lines = 150`), and `planner.py` compacts multi-step retry history into `rolling_summary`.

### Mode-Aware Per-Run Logging (`core/logger.py`)
* Console logs stream strictly to `sys.stderr` so MCP JSON-RPC `stdout` pipes are never polluted.
* Every execution of `python main.py` generates a unique timestamped log file under `logs/system/run_<YYYYMMDD_HHMMSS>.log` (or `logs/<experiment_mode>/run_<YYYYMMDD_HHMMSS>.log`).

---

## Installation & Quick Start (Windows / VS Code / CPU or GPU)

### Step 1: Unpack the Project Archive
Run `package_autosciagent.py` once to create `AutoSciAgent-LangChain.zip` and extract `./AutoSciAgent-LangChain/`:
```powershell
python package_autosciagent.py
cd AutoSciAgent-LangChain
```

### Step 2: Create the Framework Virtual Environment
```powershell
python -m venv .venv_framework
.venv_framework\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Create the Sandboxed Solver Environment (`workspace/.venv`)
`execution.py` runs generated PDE solvers inside `workspace/.venv` (automatically falling back to the active Python interpreter if `workspace/.venv` has not been created yet):
```powershell
python -m venv workspace\.venv
workspace\.venv\Scripts\pip install numpy scipy matplotlib

# Optional: Install CuPy if running on an NVIDIA CUDA GPU
# workspace\.venv\Scripts\pip install cupy-cuda12x
```

### Step 4: Configure `.env`
Open `.env` in VS Code and set your Google Gemini API key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
LOG_LEVEL=INFO
AUTOSCI_TEST_MODE=0
AUTOSCI_AUTO_APPROVE=0
```
* Set `AUTOSCI_TEST_MODE=1` to run the entire 9-node pipeline offline using the built-in deterministic responder (zero API tokens spent).
* Set `AUTOSCI_AUTO_APPROVE=1` if you want non-interactive batch runs without `[Y/n]` terminal prompts.

### Step 5: Run the Automated Test Suite
Verify the security jail, Copy-on-Write versioning, `HARNESS²` Rules 1–3 validator, all 9 individual nodes, and both branches of the closed-loop workflow (`deriver` enabled vs. `deriver` skipped):
```powershell
pytest tests/ -v
```

### Step 6: Run the 2D Heat Conduction Simulation
```powershell
python main.py
```

---

## Test Example Walkthrough: 2D Orthotropic Heat Conduction

By default, `workspace/inputs/task.md` instructs the agent to solve the 2D steady-state orthotropic heat conduction PDE on a $1.0\text{ m} \times 1.0\text{ m}$ square plate ($50 \times 50$ grid, $\Delta x = \Delta y$):

$$k_x \frac{\partial^2 T}{\partial x^2} + k_y \frac{\partial^2 T}{\partial y^2} = 0$$

using the 5-point Finite Difference stencil derived by `deriver.py`:

$$T_{i,j}^{(k+1)} = \frac{k_x \left(T_{i,j+1}^{(k)} + T_{i,j-1}^{(k)}\right) + k_y \left(T_{i+1,j}^{(k)} + T_{i-1,j}^{(k)}\right)}{2(k_x + k_y)}$$

across three boundary condition and material property scenarios:
* **Case A (Baseline Isotropic)**: Top wall $T = 100.0\text{ }^\circ\text{C}$, Bottom/Left/Right walls $T = 25.0\text{ }^\circ\text{C}$, $k_x = 1.0$, $k_y = 1.0\text{ W/(m}\cdot\text{K)}$.
* **Case B (Elevated Boundary)**: Top wall $T = 200.0\text{ }^\circ\text{C}$, Bottom/Left/Right walls $T = 25.0\text{ }^\circ\text{C}$, $k_x = 1.0$, $k_y = 1.0\text{ W/(m}\cdot\text{K)}$.
* **Case C (Anisotropic Composite)**: Top wall $T = 200.0\text{ }^\circ\text{C}$, Bottom/Left/Right walls $T = 25.0\text{ }^\circ\text{C}$, $k_x = 3.0$, $k_y = 1.0\text{ W/(m}\cdot\text{K)}$.

### Generated Deliverables on Disk
After `python main.py` completes, inspect the following files:
* `workspace/specs/simulation_spec.json`: Formal JSON specification generated by `requirement.py`.
* `workspace/derivations/derivation.md`: Discretized PDE stencil, boundary condition equations, and diagonal-dominance stability analysis generated by `deriver.py`.
* `workspace/scripts/solver_2d.py`: Standalone CPU/GPU-adaptive Python solver generated by `coder.py`.
* `workspace/outputs/data/summary_metrics.csv`: Quantitative table reporting `case_name, max_temp_c, min_temp_c, center_temp_c, iterations, residual`.
* `workspace/outputs/plots/heat_2d_comparison.png`: 4-panel figure containing 2D temperature heatmaps for Cases A, B, and C alongside the vertical centerline temperature profile $T(x=0.5\text{ m}, y)$.
* `workspace/outputs/reports/final_report.md`: Complete engineering report synthesized by `reporter.py` and audited by `judge.py`.
* `library/task_<timestamp>/EXPERIENCE.md`: Causal edit attribution table and transferable numerical lesson recorded by `reflector.py`.
* `logs/system/run_<timestamp>.log`: Full agent-level execution trace.

---

## Adapting AutoSciAgent to New PDE Problems

Because all prompts and domain skills are externalized as `.md` files, you can apply `AutoSciAgent` to new physical systems (such as transient 2D advection-diffusion, the 2D wave equation, Poisson electrostatics, or lid-driven cavity flow) without modifying any Python framework code:
* **Define the Task**: Edit `workspace/inputs/task.md` with the new governing PDE, domain geometry, boundary/initial conditions, and parameter sweep cases.
* **Add Reference Papers (Optional)**: Drop relevant `.pdf`, `.docx`, or `.md` papers into `workspace/inputs/references/`. `requirement.py` will digest them via the `reference_digester` subagent, and `judge.py` will index reusable equations into ChromaDB.
* **Add or Customize Skills (Optional)**: Create a new folder `harness/skills/<skill_name>/SKILL.md` describing the numerical scheme, and add its verbatim trigger line to `harness/guiderails.md`.
