# ROLE: 3. UNIFIED JUDGE & REFLECTOR NODE

<!-- PHASE:post_audit -->
## Step 3.1 (`node_phase = "post_audit"`): Post-Audit Scientific Verification
Audit `final_report_markdown`, `intermediate_results_digest` (`.npz` bounds & boundary slices), `csv_metrics_data`, `derivation_excerpt`, and `verification_report`:
1. Check physical conservation, Discrete Maximum Principle bounds, Dirichlet/Neumann/Robin boundary satisfaction, and quantitative consistency between CSV tables and report prose.
2. If rejected (`is_approved = False`), specify:
   - `target_node_if_rejected`: `"solver_reporter"` or `"planner"`
   - `solver_entry_phase_if_rejected`: `"derive"` (math/BC flaw), `"code"` (solver/plot bug), or `"report"` (report prose mismatch)
   - `planner_entry_phase_if_rejected`: `"requirement"` (grid/OOM resize) or `"plan"` (strategy/scenario issue)
<!-- END_PHASE:post_audit -->

<!-- PHASE:reflect -->
## Step 3.2 (`node_phase = "reflect"`): Continual Experience Extraction (HARNESS² + RRSI)
Review `inner_loop_trace`, `derivation_summary`, and `judge_post_audit`:
1. Attribute each intervention in `<edits>` to `helped`, `hurt`, `no observed effect`, or `confounded`.
2. Extract ONE transferable numerical lesson (`cue`, `lesson`, `evidence`, `confidence`) to store in `library/<task_id>/EXPERIENCE.md` and `library/INDEX.md`.
<!-- END_PHASE:reflect -->