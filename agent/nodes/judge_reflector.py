# agent/nodes/judge_reflector.py
import json
import shutil
from core import PROJECT_ROOT, get_llm, load_config
from core.harness_manager import HarnessManager
from core.security import get_active_workspace
from core.tools_hitl import request_human_approval
from core.tools_io import smart_read_file
from core.logger import setup_logger
from agent.state import AutoSciState, JudgePostAuditOutput, ReflectionOutput

logger = setup_logger("Node-JudgeReflector", mode="system")


def judge_reflector_node(state: AutoSciState) -> dict:
    """
    3. UNIFIED JUDGE & REFLECTOR NODE:
    - Step 3.1 (Post-Audit Scientific Verification):
        * Audits saved `final_report.md`, `intermediate_results_digest` (`.npz` bounds & boundary slices),
          CSV metrics, and plots for physical conservation, Discrete Maximum Principle, and consistency.
        * If rejected and `outer_loop_count < max_outer_loops`, routes to `solver_reporter`
          (`entry = "derive" | "code" | "report"`) or `planner` (`entry = "requirement" | "plan"`).
        * If rejected and `outer_loop_count >= max_outer_loops` (Circuit Breaker), skips Step 3.2 to prevent
          library poisoning and exits cleanly.
    - Step 3.2 (Continual Experience Extraction - HARNESS² + RRSI):
        * Reads `inner_loop_trace`, attributes `<edits>`, extracts transferable lesson (`cue`, `lesson`, `evidence`),
          validates HARNESS² Rules 1-3 & RRSI cap, and uses a single atomic HITL gate to save
          `library/<task_id>/EXPERIENCE.md`, `skills_snapshot/`, and `library/INDEX.md`.
    """
    cfg = load_config()
    agent_cfg = cfg["agent"]
    paths_cfg = cfg["paths"]
    det_temp = float(cfg["llm"]["deterministic_temperature"])
    max_outer = int(agent_cfg["max_outer_loops"])
    preview_lines = int(agent_cfg["max_log_tail_lines"])
    lib_rel = paths_cfg["library_dir"]

    hm = HarnessManager()
    llm_det = get_llm(temperature=det_temp)

    active_ws = get_active_workspace()
    task_id = state.get("task_id") or active_ws.name
    spec = state.get("simulation_spec", {})
    verify_rep = state.get("verification_report", {"ok": False})
    results_digest = state.get("intermediate_results_digest", {})

    # Verify final_report.md exists on disk
    report_rel_path = state.get("report_path", paths_cfg["report_relative_path"])
    report_disk_path = active_ws / report_rel_path
    report_md = (
        report_disk_path.read_text(encoding="utf-8")
        if report_disk_path.exists()
        else state.get("final_report_markdown", "")
    )

    data_previews = "\n\n".join(smart_read_file(df, 1, preview_lines) for df in state.get("generated_data", []))
    derivation_preview = (
        smart_read_file(state["derivation_path"], 1, preview_lines * 2)
        if state.get("derivation_path") and (active_ws / state["derivation_path"]).exists()
        else "Derivation skipped."
    )

    # ------------------------------------------------------------------
    # STEP 3.1: POST-AUDIT SCIENTIFIC VERIFICATION
    # ------------------------------------------------------------------
    audit_prompt = hm.load_node_prompt(
        "judge_reflector",
        {
            "node_phase": "post_audit",
            "task_id": task_id,
            "simulation_spec": json.dumps(spec, indent=2),
            "derivation_excerpt": derivation_preview,
            "intermediate_results_digest": json.dumps(results_digest, indent=2),
            "csv_metrics_data": data_previews,
            "verification_report": json.dumps(verify_rep, indent=2),
            "generated_plots": str(state.get("generated_plots", [])),
            "final_report_markdown": report_md,
        },
    )
    post_out: JudgePostAuditOutput = llm_det.with_structured_output(JudgePostAuditOutput).invoke(audit_prompt)

    outer = state.get("outer_loop_count", 0)
    if not post_out.is_approved:
        if outer < max_outer:
            target = post_out.target_node_if_rejected
            solver_entry = post_out.solver_entry_phase_if_rejected
            planner_entry = post_out.planner_entry_phase_if_rejected
            logger.warning(
                f"[{task_id}] Judge Post-Audit rejected results: {post_out.feedback_if_rejected} "
                f"-> Routing to [{target.upper()}] (solver_entry={solver_entry}, planner_entry={planner_entry})."
            )
            return {
                "judge_feedback": post_out.feedback_if_rejected,
                "diagnostic_feedback": post_out.feedback_if_rejected,
                "solver_entry_phase": solver_entry,
                "planner_entry_phase": planner_entry,
                "needs_derivation": (solver_entry == "derive"),
                "outer_loop_count": outer + 1,
                "inner_retry_count": 0,
                "next_node": target,
            }

        logger.warning(
            f"[{task_id}] Judge Post-Audit rejected results at max outer loops ({outer}); "
            "skipping Step 3.2 library update to prevent library poisoning."
        )
        return {
            "judge_feedback": post_out.feedback_if_rejected,
            "next_node": "end",
        }

    if not verify_rep.get("ok", False):
        logger.warning(f"[{task_id}] Verification report not OK; skipping Step 3.2 library update.")
        return {
            "judge_feedback": post_out.physics_and_logic_audit,
            "next_node": "end",
        }

    logger.info(f"[{task_id}] Judge Post-Audit APPROVED scientific package: {post_out.physics_and_logic_audit}")

    # ------------------------------------------------------------------
    # STEP 3.2: CONTINUAL EXPERIENCE EXTRACTION (HARNESS² + RRSI)
    # ------------------------------------------------------------------
    is_valid, val_errors = hm.validate_harness_tree()
    if not is_valid:
        logger.warning(f"[{task_id}] HARNESS² / RRSI validation failed ({val_errors}); skipping library write.")
        return {
            "judge_feedback": post_out.physics_and_logic_audit,
            "next_node": "end",
        }

    reflect_prompt = hm.load_node_prompt(
        "judge_reflector",
        {
            "node_phase": "reflect",
            "task_id": task_id,
            "simulation_spec": json.dumps(spec, indent=2),
            "rolling_summary": state.get("rolling_summary", ""),
            "derivation_summary": state.get("derivation_summary", ""),
            "inner_loop_trace": json.dumps(state.get("inner_loop_trace", []), indent=2),
            "judge_post_audit": post_out.physics_and_logic_audit,
            "harness_validation_status": f"valid={is_valid}, errors={val_errors}",
        },
    )
    ref_out: ReflectionOutput = llm_det.with_structured_output(ReflectionOutput).invoke(reflect_prompt)

    if not ref_out.store_entry:
        return {
            "judge_feedback": post_out.physics_and_logic_audit,
            "next_node": "end",
        }

    lib_root = (PROJECT_ROOT / lib_rel).resolve()
    entry_dir = lib_root / task_id
    entry_rel = f"{lib_rel}/{task_id}/EXPERIENCE.md"

    if not request_human_approval(
        "UPDATE CONTINUAL LIBRARY (OUTSIDE WORKSPACE)",
        f"Save experience '{entry_rel}', skills_snapshot/, and update '{lib_rel}/INDEX.md'",
    ):
        logger.info(f"[{task_id}] Operator declined library update; exiting cleanly.")
        return {
            "judge_feedback": post_out.physics_and_logic_audit,
            "next_node": "end",
        }

    edits_lines = "\n".join(
        f"- **Component**: `{e.component}` | **Verdict**: `{e.verdict}`\n"
        f"  - *Change*: {e.change}\n"
        f"  - *Observed*: {e.observed}"
        for e in ref_out.edits
    )
    exp_md = (
        f"# HARNESS² & RRSI Experience Record (`{task_id}`)\n\n"
        f"## Edit Attribution (`<edits>`)\n{edits_lines or 'No isolated edits.'}\n\n"
        f"## Transferable Lesson (`<experience>`)\n"
        f"- **Cue**: {ref_out.cue}\n"
        f"- **Lesson**: {ref_out.lesson}\n"
        f"- **Evidence**: {ref_out.evidence}\n"
        f"- **Confidence**: {ref_out.confidence}\n"
    )

    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "EXPERIENCE.md").write_text(exp_md, encoding="utf-8")

    skills_src = hm.harness_dir / "skills"
    skills_dst = entry_dir / "skills_snapshot"
    if skills_src.exists():
        shutil.copytree(skills_src, skills_dst, dirs_exist_ok=True)

    index_path = lib_root / "INDEX.md"
    existing_idx = (
        index_path.read_text(encoding="utf-8")
        if index_path.exists()
        else "# AutoSciAgent Harness Improvement Library Index\n\n"
    )
    new_line = f"- **[`{task_id}`]({task_id}/EXPERIENCE.md)** — **Cue:** {ref_out.cue} | **Lesson:** {ref_out.lesson}\n"
    if f"[`{task_id}`]" not in existing_idx:
        index_path.write_text(existing_idx + new_line, encoding="utf-8")

    logger.info(f"[{task_id}] Saved experience record to '{entry_rel}' and updated '{lib_rel}/INDEX.md'.")
    return {
        "judge_feedback": post_out.physics_and_logic_audit,
        "library_entry_path": entry_rel,
        "next_node": "end",
    }