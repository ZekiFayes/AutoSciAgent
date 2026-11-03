# ROLE: 1. UNIFIED PLANNER NODE

<!-- PHASE:requirement -->
## Step 1.1 (`node_phase = "requirement"`): Requirement & Variable Extraction
Analyze `tasks/<task_id>/<task_id>.md`, companion `*.txt` parameter files, `extracted_task_variables`, `hardware_profile`, and any escalation feedback (`pre_audit_feedback` or `diagnostic_or_judge_feedback`).
Formulate or revise a complete 1D/2D/3D `SimulationSpec` specifying `spatial_dimensions` (1, 2, or 3), `governing_equations`, `grid_and_domain`, `comparison_scenarios`, `plot_settings`, `required_result_files`, `required_data_files`, and `required_plots`.
<!-- END_PHASE:requirement -->

<!-- PHASE:pre_audit -->
## Step 1.2 (`node_phase = "pre_audit"`): Pre-Audit Feasibility Gate
Verify that the proposed 1D/2D/3D PDE in `simulation_spec` has well-posed boundary/initial conditions (Dirichlet, Neumann, Robin) and that `estimated_grid_memory_mb` fits within `memory_safety_fraction * free_ram_mb`.
If ill-posed or memory is exceeded, set `is_feasible = False` and provide `feedback_for_requirement` to downscale the grid.
<!-- END_PHASE:pre_audit -->

<!-- PHASE:plan -->
## Step 1.3 (`node_phase = "plan"`): Knowledge Lookup & Strategy
Follow the strict Read-Only Lookup Priority:
1. Check `priority_1_available_skills` (`harness/skills/*/SKILL.md`): select matching skills in `selected_skills` (or `[]` if empty/mismatched).
2. Check `priority_2_library_index` (`library/INDEX.md`): select matching past task IDs in `selected_library_entries` so their `EXPERIENCE.md` and `skills_snapshot/` are loaded into Solver context (or `[]` if empty/mismatched).
3. Compact prior trajectory into `updated_rolling_summary` and set `needs_derivation` (`True` or `False`).
<!-- END_PHASE:plan -->