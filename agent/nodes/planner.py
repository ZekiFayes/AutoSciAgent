# agent/nodes/planner.py
import json
from pathlib import Path
from core import PROJECT_ROOT, get_llm, load_config
from core.document_parser import load_task_variables_from_folder, parse_document
from core.hardware import (
    clamp_grid_to_memory_limit,
    detect_hardware_profile,
    estimate_grid_memory_mb,
)
from core.harness_manager import HarnessManager
from core.security import TASKS_DIR, get_active_workspace, set_active_workspace
from core.tools_io import smart_write_file
from core.logger import setup_logger
from agent.state import AutoSciState, PlanOutput, PreAuditOutput, SimulationSpec

logger = setup_logger("Node-Planner", mode="system")


def planner_node(state: AutoSciState) -> dict:
    """
    1. UNIFIED PLANNER NODE:
    - Entry Router:
        * If `simulation_spec` is missing OR `planner_entry_phase == "requirement"`, runs Step 1.1 & Step 1.2.
        * Otherwise (`planner_entry_phase == "plan"`), skips directly to Step 1.3.
    - Step 1.1 (Requirement & Variable Extraction): Reads `tasks/<task_id>/<task_id>.md` & `*.txt` into variables.
    - Step 1.2 (Pre-Audit Feasibility Gate): Verifies PDE well-posedness and checks RAM limits (`req_retry < max_req`),
      deterministically clamping grid dimensions if retries are exhausted before saving `specs/simulation_spec.json`.
    - Step 1.3 (Knowledge Lookup & Strategy): Checks `1st: harness/skills/*/SKILL.md` -> `2nd: library/INDEX.md`
      (loading matched `EXPERIENCE.md` and `skills_snapshot/`), skipping cleanly if empty or mismatched.
    """
    cfg = load_config()
    agent_cfg = cfg["agent"]
    paths_cfg = cfg["paths"]

    gen_temp = float(cfg["llm"]["temperature"])
    det_temp = float(cfg["llm"]["deterministic_temperature"])
    max_req_retries = int(agent_cfg["max_req_retries"])
    mem_safety_fraction = float(agent_cfg["memory_safety_fraction"])
    skill_preview_chars = int(agent_cfg["max_skill_preview_chars"])
    spec_rel_path = paths_cfg["spec_relative_path"]

    hm = HarnessManager()
    llm_gen = get_llm(temperature=gen_temp)
    llm_det = get_llm(temperature=det_temp)

    rel_task = state.get("task_file_path") or "tasks/task_01/task_01.md"
    task_path = (Path(rel_task) if Path(rel_task).is_absolute() else (PROJECT_ROOT / rel_task)).resolve()
    task_id = state.get("task_id") or (task_path.parent.name if task_path.parent != TASKS_DIR else task_path.stem)

    set_active_workspace(task_id)
    active_ws = get_active_workspace()

    hw_profile = state.get("hardware_profile") or detect_hardware_profile()
    free_mb = float(hw_profile.get("free_ram_mb", cfg["hardware_defaults"]["free_ram_mb"]))
    max_allowed_mb = round(mem_safety_fraction * free_mb, 4)

    raw_task = state.get("raw_task_text") or parse_document(task_path)
    extracted_vars = state.get("extracted_variables") or load_task_variables_from_folder(task_path)

    entry_phase = state.get("planner_entry_phase", "requirement")
    spec_dict = dict(state.get("simulation_spec") or {})
    escalation_feedback = state.get("diagnostic_feedback") or state.get("judge_feedback") or ""

    # ------------------------------------------------------------------
    # STEP 1.1 & STEP 1.2: REQUIREMENT EXTRACTION & PRE-AUDIT GATE
    # ------------------------------------------------------------------
    if not spec_dict or entry_phase == "requirement":
        pre_audit_feedback = escalation_feedback
        for req_retry in range(max_req_retries + 1):
            req_prompt = hm.load_node_prompt(
                "planner",
                {
                    "node_phase": "requirement",
                    "task_id": task_id,
                    "task_markdown": raw_task,
                    "extracted_task_variables": json.dumps(extracted_vars, indent=2),
                    "hardware_profile": json.dumps(hw_profile, indent=2),
                    "pre_audit_feedback": pre_audit_feedback,
                },
            )
            spec_obj: SimulationSpec = llm_det.with_structured_output(SimulationSpec).invoke(req_prompt)
            spec_dict = spec_obj.model_dump()

            for key in (
                "task_title",
                "domain",
                "spatial_dimensions",
                "governing_equations",
                "needs_derivation",
                "grid_and_domain",
                "comparison_scenarios",
                "plot_settings",
                "required_result_files",
                "required_data_files",
                "required_plots",
            ):
                if key in extracted_vars and extracted_vars[key] is not None:
                    spec_dict[key] = extracted_vars[key]

            dims = int(spec_dict.get("spatial_dimensions", 2))
            est_mb = estimate_grid_memory_mb(spec_dict.get("grid_and_domain", {}), spatial_dims=dims)

            audit_prompt = hm.load_node_prompt(
                "planner",
                {
                    "node_phase": "pre_audit",
                    "task_id": task_id,
                    "simulation_spec": json.dumps(spec_dict, indent=2),
                    "estimated_grid_memory_mb": f"{est_mb} MB (Max Allowed: {max_allowed_mb} MB)",
                },
            )
            audit_out: PreAuditOutput = llm_det.with_structured_output(PreAuditOutput).invoke(audit_prompt)

            if audit_out.is_feasible and est_mb <= max_allowed_mb:
                logger.info(f"[{task_id}] Planner Pre-Audit PASSED ({est_mb} MB <= {max_allowed_mb} MB).")
                break

            if req_retry < max_req_retries:
                pre_audit_feedback = (
                    audit_out.feedback_for_requirement
                    or audit_out.feasibility_notes
                    or f"Estimated grid memory {est_mb} MB exceeds limit {max_allowed_mb} MB; reduce nx, ny, nz."
                )
                logger.warning(f"[{task_id}] Planner Pre-Audit retry {req_retry + 1}: {pre_audit_feedback}")
            else:
                spec_dict["grid_and_domain"] = clamp_grid_to_memory_limit(
                    spec_dict.get("grid_and_domain", {}), spatial_dims=dims, max_allowed_mb=max_allowed_mb
                )
                logger.warning(
                    f"[{task_id}] Planner Pre-Audit retries exhausted; deterministically clamped grid to "
                    f"{spec_dict['grid_and_domain']}."
                )

        smart_write_file(spec_rel_path, json.dumps(spec_dict, indent=2), author_node="planner_spec")
        logger.info(
            f"[{task_id}] Saved validated {spec_dict.get('spatial_dimensions')}D SimulationSpec to "
            f"workspace/{active_ws.name}/{spec_rel_path}"
        )

    # ------------------------------------------------------------------
    # STEP 1.3: STRICT PRIORITY KNOWLEDGE LOOKUP & STRATEGY
    # ------------------------------------------------------------------
    available_skills = hm.list_available_skills()
    if available_skills:
        skills_summary = "\n\n".join(
            f"### Skill `{k}`\n{v[:skill_preview_chars]}" for k, v in available_skills.items()
        )
        logger.info(f"[{task_id}] Planner Lookup (1st Priority): Found skills {list(available_skills.keys())}.")
    else:
        skills_summary = "EMPTY (Skip SKILL.md)"
        logger.info(f"[{task_id}] Planner Lookup (1st Priority): `harness/skills/` is empty -> skipping.")

    library_index = hm.load_library_index()
    available_lib_entries = set(hm.list_library_entries())
    if library_index and available_lib_entries:
        lib_summary = f"Available Library Entries {sorted(available_lib_entries)}:\n{library_index}"
        logger.info(f"[{task_id}] Planner Lookup (2nd Priority): Found library entries {sorted(available_lib_entries)}.")
    else:
        lib_summary = "EMPTY (Skip library/INDEX.md)"
        logger.info(f"[{task_id}] Planner Lookup (2nd Priority): `library/INDEX.md` is empty -> skipping.")

    plan_prompt = hm.load_node_prompt(
        "planner",
        {
            "node_phase": "plan",
            "task_id": task_id,
            "simulation_spec": json.dumps(spec_dict, indent=2),
            "hardware_profile": json.dumps(hw_profile, indent=2),
            "priority_1_available_skills": skills_summary,
            "priority_2_library_index": lib_summary,
            "previous_rolling_summary": state.get("rolling_summary", "Initial planning phase."),
            "diagnostic_or_judge_feedback": escalation_feedback or "None",
        },
    )
    plan_out: PlanOutput = llm_gen.with_structured_output(PlanOutput).invoke(plan_prompt)

    valid_skills = [s for s in plan_out.selected_skills if s in available_skills]
    valid_lib_entries = [e for e in plan_out.selected_library_entries if e in available_lib_entries and e != task_id]

    if "needs_derivation" in state and state["needs_derivation"] is False:
        needs_deriv = False
    else:
        needs_deriv = bool(plan_out.needs_derivation and spec_dict.get("needs_derivation", True))

    solver_entry: str = "derive" if needs_deriv else "code"
    logger.info(
        f"[{task_id}] Planner completed strategy (skills={valid_skills}, "
        f"library_entries={valid_lib_entries}, solver_entry_phase={solver_entry}) -> [SOLVER_REPORTER]."
    )

    return {
        "task_id": task_id,
        "raw_task_text": raw_task,
        "extracted_variables": extracted_vars,
        "hardware_profile": hw_profile,
        "simulation_spec": spec_dict,
        "plan_steps": plan_out.plan_steps,
        "needs_derivation": needs_deriv,
        "active_skills": valid_skills,
        "active_library_entries": valid_lib_entries,
        "rolling_summary": plan_out.updated_rolling_summary,
        "planner_directive": plan_out.directive_for_next_node,
        "solver_entry_phase": solver_entry,
        "inner_retry_count": 0,
        "next_node": "solver_reporter",
    }