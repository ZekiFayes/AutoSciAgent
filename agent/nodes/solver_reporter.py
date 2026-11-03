# agent/nodes/solver_reporter.py
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from core import PROJECT_ROOT, get_llm, load_config, load_solver_template_code
from core.harness_manager import HarnessManager
from core.security import classify_and_resolve_path, get_active_workspace
from core.tools_hitl import request_human_approval
from core.tools_io import archive_and_clear_outputs_cow, smart_read_file, smart_write_file
from core.tools_search import mcp_inspect_results, mcp_search_workspace
from core.tools_io import mcp_read_file
from core.logger import setup_logger
from harness.plugins.runtime_middleware import apply_post_execution_middleware
from harness.scripts.inspect_results import inspect_workspace_results
from harness.scripts.verify_outputs import verify_workspace_outputs
from agent.state import (
    AutoSciState,
    CodeGenerationOutput,
    DerivationAuditOutput,
    DerivationOutput,
    DiagnoserOutput,
    ReporterOutput,
)

logger = setup_logger("Node-SolverReporter", mode="system")


def get_shared_python() -> str:
    cfg = load_config()
    venv_rel = cfg["paths"]["shared_venv_dir"]
    shared_venv = (PROJECT_ROOT / venv_rel).resolve()

    for cand in (
        shared_venv / "Scripts" / "python.exe",
        shared_venv / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


def _find_missing_packages(python_exe: str, packages: list[str]) -> list[str]:
    missing = []
    alias_map = {"scikit-learn": "sklearn", "pyyaml": "yaml", "pillow": "PIL", "opencv-python": "cv2"}
    for pkg in packages:
        clean_pkg = pkg.split(">=")[0].split("==")[0].split("<")[0].strip()
        if not clean_pkg:
            continue
        mod_name = alias_map.get(clean_pkg.lower(), clean_pkg.replace("-", "_"))
        check = subprocess.run([python_exe, "-c", f"import {mod_name}"], capture_output=True, text=True)
        if check.returncode != 0:
            missing.append(clean_pkg)
    return missing


def _run_stencil_checker_if_available(hm: HarnessManager, python_exe: str) -> str:
    checker = hm.harness_dir / "skills" / "numerical_pde" / "scripts" / "check_stencil.py"
    if not checker.exists():
        return "No companion stencil checker script."
    try:
        proc = subprocess.run([python_exe, str(checker)], capture_output=True, text=True, timeout=10)
        return (proc.stdout or proc.stderr).strip()
    except Exception as e:
        return f"Stencil check skipped: {e}"


def _ensure_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


def solver_reporter_node(state: AutoSciState) -> dict:
    """
    2. UNIFIED SOLVER & REPORTER NODE:
    - Entry Router (resets `attempt = 0`, `deriv_retry = 0`):
        * `solver_entry_phase == "derive"` -> Step 2.1 (Derive) -> Step 2.2 (Derivation Audit) -> Step 2.3 (Code & Debug)
        * `solver_entry_phase == "code"`   -> Step 2.3 (Code & Debug)
        * `solver_entry_phase == "report"` -> Step 2.6 (Report Synthesis)
    - Step 2.3 (Code & Debug Phase): Injects loaded `SKILL.md` + `EXPERIENCE.md` priors, full `stderr_traceback`,
      `intermediate_results_digest` (from `inspect_results.py`, with fallback to `.versions/*.npz`), full existing
      `scripts/solver.py` (up to 600 lines), stepped retry temperature, and SHA-256 stagnation detection.
    - Step 2.4 (Execute & Verify Phase): Moves prior `outputs/{results,data,plots}/*` to `.versions/` via
      `archive_and_clear_outputs_cow()`, runs `solver.py` autonomously, and runs `verify_outputs.py` + `inspect_results.py`.
    - Step 2.6 (Report Synthesis Phase): Cross-checks deliverables against `SimulationSpec`, writes
      `outputs/reports/final_report.md`, and verifies its existence on disk.
    """
    cfg = load_config()
    llm_cfg = cfg["llm"]
    agent_cfg = cfg["agent"]
    paths_cfg = cfg["paths"]

    base_temp = float(llm_cfg["temperature"])
    temp_step = float(llm_cfg["retry_temperature_step"])
    max_retry_temp = float(llm_cfg["max_retry_temperature"])
    det_temp = float(llm_cfg["deterministic_temperature"])

    max_deriv_retries = int(agent_cfg["max_deriv_retries"])
    max_inner_retries = int(agent_cfg["max_inner_retries"])
    max_outer_loops = int(agent_cfg["max_outer_loops"])
    timeout_sec = int(agent_cfg["execution_timeout_sec"])
    max_tail_lines = int(agent_cfg["max_log_tail_lines"])
    max_tb_lines = int(agent_cfg["max_traceback_lines"])
    max_solver_lines = int(agent_cfg["max_solver_code_lines"])
    min_png_bytes = int(agent_cfg["min_valid_png_bytes"])

    rel_deriv_path = paths_cfg["derivation_relative_path"]
    default_script_path = paths_cfg["solver_relative_path"]
    spec_rel_path = paths_cfg["spec_relative_path"]
    rel_report_path = paths_cfg["report_relative_path"]

    hm = HarnessManager()
    llm_det = get_llm(temperature=det_temp)
    python_exe = get_shared_python()

    active_ws = get_active_workspace()
    task_id = state.get("task_id") or active_ws.name
    spec = state.get("simulation_spec", {})
    dims = int(spec.get("spatial_dimensions", 2))

    # Load Priority 1 (SKILL.md) and Priority 2 (EXPERIENCE.md + skills_snapshot) priors
    active_skills = state.get("active_skills", [])
    skills_text = "\n\n".join(filter(None, (hm.load_skill(s) for s in active_skills))) or "Empty (Skipped)"

    active_lib_entries = state.get("active_library_entries", [])
    lib_experiences_text = (
        "\n\n".join(filter(None, (hm.load_library_experience(eid) for eid in active_lib_entries)))
        or "Empty (Skipped)"
    )

    entry_phase = state.get("solver_entry_phase")
    if entry_phase not in ("derive", "code", "report"):
        entry_phase = "derive" if state.get("needs_derivation", True) else "code"

    should_derive = (entry_phase == "derive")
    skip_to_report = (entry_phase == "report")

    derivation_path = state.get("derivation_path", "")
    derivation_summary = state.get("derivation_summary", "")
    derivation_audit_notes = state.get("derivation_audit_notes", "")
    script_path = state.get("generated_code_path", default_script_path)
    diagnostic_feedback = state.get("diagnostic_feedback") or state.get("judge_feedback") or "None"
    judge_feedback = state.get("judge_feedback", "None")

    exec_status = state.get("execution_status", "not_run")
    rel_log_path = state.get("log_file_path", "")
    log_tail = state.get("log_tail", "")
    stderr_traceback = state.get("stderr_traceback", "")
    verify_rep: dict = state.get("verification_report", {"ok": False, "issues": []})
    results_digest: dict = state.get("intermediate_results_digest") or inspect_workspace_results(active_ws)
    results_files: list[str] = list(state.get("generated_results", []))
    plots: list[str] = list(state.get("generated_plots", []))
    data_files: list[str] = list(state.get("generated_data", []))
    required_packages: list[str] = list(state.get("required_packages", []))
    inner_loop_trace: list[dict] = list(state.get("inner_loop_trace", []))

    prev_code_hash = ""
    existing_script_file = active_ws / script_path
    if existing_script_file.exists():
        prev_code_hash = hashlib.sha256(existing_script_file.read_bytes()).hexdigest()[:12]

    attempt = 0
    while attempt < max_inner_retries:
        retry_temp = min(max_retry_temp, round(base_temp + temp_step * attempt, 3))
        llm_gen = get_llm(temperature=retry_temp)

        if not skip_to_report:
            # ------------------------------------------------------------------
            # STEP 2.1 & STEP 2.2: DERIVE PHASE + DERIVATION AUDIT GATE
            # ------------------------------------------------------------------
            if should_derive:
                deriv_feedback = ""
                for deriv_retry in range(max_deriv_retries + 1):
                    derive_prompt = hm.load_node_prompt(
                        "solver_reporter",
                        {
                            "node_phase": "derive",
                            "task_id": task_id,
                            "spatial_dimensions": f"{dims}D",
                            "simulation_spec": json.dumps(spec, indent=2),
                            "planner_directive": state.get("planner_directive", ""),
                            "loaded_skills": skills_text,
                            "loaded_library_experiences": lib_experiences_text,
                            "derivation_audit_feedback": deriv_feedback or "None",
                            "diagnostic_error_feedback": diagnostic_feedback,
                            "intermediate_results_digest": json.dumps(results_digest, indent=2),
                        },
                    )
                    deriv_out: DerivationOutput = llm_gen.with_structured_output(DerivationOutput).invoke(derive_prompt)
                    full_md = (
                        f"{deriv_out.derivation_markdown}\n\n"
                        f"## Stability & Convergence Notes\n{deriv_out.stability_and_convergence_notes}\n\n"
                        f"## Discretized Stencil Summary\n`{deriv_out.discretized_stencil_summary}`\n"
                    )
                    write_res = smart_write_file(rel_deriv_path, full_md, author_node="solver_deriver")
                    derivation_path = rel_deriv_path
                    derivation_summary = deriv_out.discretized_stencil_summary
                    logger.info(
                        f"[{task_id}] Solver (Step 2.1 Derive, Attempt {attempt + 1}, DerivRetry {deriv_retry}): {write_res}"
                    )

                    # Step 2.2: Derivation Audit Gate
                    stencil_check_msg = _run_stencil_checker_if_available(hm, python_exe)
                    audit_deriv_prompt = hm.load_node_prompt(
                        "solver_reporter",
                        {
                            "node_phase": "audit_derivation",
                            "task_id": task_id,
                            "simulation_spec": json.dumps(spec, indent=2),
                            "derivation_markdown": full_md,
                            "stencil_check_output": stencil_check_msg,
                        },
                    )
                    audit_deriv_out: DerivationAuditOutput = llm_det.with_structured_output(
                        DerivationAuditOutput
                    ).invoke(audit_deriv_prompt)
                    derivation_audit_notes = audit_deriv_out.math_and_bc_audit

                    if audit_deriv_out.is_valid:
                        logger.info(f"[{task_id}] Solver (Step 2.2 Derivation Audit): PASSED ({derivation_audit_notes}).")
                        break

                    if deriv_retry < max_deriv_retries:
                        deriv_feedback = audit_deriv_out.feedback_for_deriver or audit_deriv_out.math_and_bc_audit
                        logger.warning(
                            f"[{task_id}] Solver (Step 2.2 Derivation Audit) rejected math -> looping to Step 2.1: {deriv_feedback}"
                        )
                    else:
                        derivation_audit_notes = f"WARNING (Audit retries exhausted): {audit_deriv_out.math_and_bc_audit}"
                        logger.warning(f"[{task_id}] {derivation_audit_notes} -> Proceeding to Step 2.3.")

            # ------------------------------------------------------------------
            # STEP 2.3: CODE & DEBUG PHASE (With Log & Intermediate Result Access)
            # ------------------------------------------------------------------
            deriv_text = (
                smart_read_file(derivation_path, start_line=1, max_lines=max_solver_lines)
                if derivation_path and (active_ws / derivation_path).exists()
                else "No separate derivation file (direct implementation)."
            )

            # Refresh intermediate results digest (reads outputs/results/*.npz or falls back to .versions/*.npz)
            results_digest = inspect_workspace_results(active_ws)
            prev_code = ""
            if (active_ws / script_path).exists() and (attempt > 0 or diagnostic_feedback != "None"):
                prev_code = smart_read_file(script_path, start_line=1, max_lines=max_solver_lines)

            available_debug_tools = [t.name for t in (mcp_read_file, mcp_inspect_results, mcp_search_workspace)]

            code_prompt = hm.load_node_prompt(
                "solver_reporter",
                {
                    "node_phase": "code",
                    "task_id": task_id,
                    "attempt_number": f"{attempt + 1} of {max_inner_retries} (temperature={retry_temp})",
                    "available_debug_inspection_tools": str(available_debug_tools),
                    "target_script_path": default_script_path,
                    "spec_file_to_load_variables_from": spec_rel_path,
                    "spatial_dimensions": f"{dims}D",
                    "simulation_spec": json.dumps(spec, indent=2),
                    "reference_solver_template": load_solver_template_code(),
                    "hardware_profile": json.dumps(state.get("hardware_profile", {}), indent=2),
                    "planner_directive": state.get("planner_directive", ""),
                    "mathematical_derivation": deriv_text,
                    "derivation_audit_notes": derivation_audit_notes,
                    "loaded_skills": skills_text,
                    "loaded_library_experiences": lib_experiences_text,
                    "diagnostic_error_feedback": diagnostic_feedback,
                    "judge_feedback": judge_feedback,
                    "stderr_traceback": stderr_traceback or "No prior stderr traceback.",
                    "last_execution_log_tail": log_tail or "No prior execution log.",
                    "intermediate_results_digest": json.dumps(results_digest, indent=2),
                    "last_verification_issues": json.dumps(verify_rep.get("issues", [])),
                    "inner_loop_trace": json.dumps(inner_loop_trace, indent=2) if inner_loop_trace else "First attempt",
                    "previous_code_if_retrying": prev_code,
                },
            )
            code_out: CodeGenerationOutput = llm_gen.with_structured_output(CodeGenerationOutput).invoke(code_prompt)
            script_path = code_out.script_relative_path or default_script_path
            if not script_path.startswith("scripts/"):
                script_path = f"scripts/{script_path}"
            required_packages = code_out.required_pip_packages

            new_code_hash = hashlib.sha256(code_out.python_code.encode("utf-8")).hexdigest()[:12]
            stagnation_detected = bool(prev_code and new_code_hash == prev_code_hash)
            prev_code_hash = new_code_hash

            write_msg = smart_write_file(script_path, code_out.python_code, author_node="solver_coder")
            logger.info(
                f"[{task_id}] Solver (Step 2.3 Code & Debug, Attempt {attempt + 1}, sha256={new_code_hash}): {write_msg}"
            )

            # ------------------------------------------------------------------
            # STEP 2.4: AUTONOMOUS EXECUTE & VERIFY PHASE (Clean-Slate Move Guard)
            # ------------------------------------------------------------------
            safe_script, _ = classify_and_resolve_path(
                script_path, allow_outside_workspace=False, write_operation=False
            )
            log_dir = active_ws / "outputs" / "run_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = log_dir / f"run_{ts}.log"
            rel_log_path = str(log_file.relative_to(active_ws)).replace("\\", "/")

            missing_pkgs = _find_missing_packages(python_exe, required_packages)
            if missing_pkgs:
                if request_human_approval(
                    "OUTSIDE-WORKSPACE PACKAGE INSTALL (SHARED .VENV)",
                    f"Install {missing_pkgs} into shared environment '{python_exe}'",
                ):
                    subprocess.run(
                        [python_exe, "-m", "pip", "install", *missing_pkgs], cwd=str(PROJECT_ROOT), check=False
                    )

            if not safe_script.exists():
                raw_log = f"Execution Error: Script 'workspace/{task_id}/{script_path}' does not exist."
                stderr_traceback = raw_log
                exec_status = "failed"
            else:
                # Move prior outputs to .versions/ so stale files never cause false-positives
                archive_and_clear_outputs_cow(author_node="solver_pre_exec")
                try:
                    proc = subprocess.run(
                        [python_exe, str(safe_script)],
                        cwd=str(active_ws),
                        capture_output=True,
                        text=True,
                        timeout=timeout_sec,
                    )
                    raw_log = (
                        f"=== RETURN CODE: {proc.returncode} ===\n"
                        f"--- STDOUT ---\n{proc.stdout}\n"
                        f"--- STDERR ---\n{proc.stderr}\n"
                    )
                    stderr_lines = (proc.stderr or "").strip().splitlines()
                    stderr_traceback = "\n".join(stderr_lines[-max_tb_lines:]) if stderr_lines else ""
                    exec_status = "success" if proc.returncode == 0 else "failed"
                except subprocess.TimeoutExpired as e:
                    raw_log = f"TIMEOUT ERROR after {timeout_sec}s.\nSTDOUT:\n{e.stdout or ''}\nSTDERR:\n{e.stderr or ''}\n"
                    stderr_traceback = f"TIMEOUT ERROR after {timeout_sec}s."
                    exec_status = "timeout"

            verify_rep = verify_workspace_outputs(active_ws, min_png_bytes=min_png_bytes)
            results_digest = inspect_workspace_results(active_ws)

            results_dir = active_ws / "outputs" / "results"
            plots_dir = active_ws / "outputs" / "plots"
            data_dir = active_ws / "outputs" / "data"

            results_files = (
                [str(p.relative_to(active_ws)).replace("\\", "/") for p in sorted(results_dir.glob("*.*"))]
                if results_dir.exists()
                else []
            )
            plots = (
                [str(p.relative_to(active_ws)).replace("\\", "/") for p in sorted(plots_dir.glob("*.png"))]
                if plots_dir.exists()
                else []
            )
            data_files = (
                [str(p.relative_to(active_ws)).replace("\\", "/") for p in sorted(data_dir.glob("*.*"))]
                if data_dir.exists()
                else []
            )

            processed_log = apply_post_execution_middleware(
                raw_log=raw_log,
                retry_count=attempt,
                results_found=results_files,
                plots_found=plots,
                data_found=data_files,
            )
            full_log = f"{processed_log}\n\n[VERIFICATION_REPORT]\n{json.dumps(verify_rep, indent=2)}\n"
            log_file.write_text(full_log, encoding="utf-8")

            lines = processed_log.splitlines()
            log_tail = "\n".join(lines[-max_tail_lines:]) if len(lines) > max_tail_lines else processed_log
            logger.info(
                f"[{task_id}] Solver (Step 2.4 Execute, Attempt {attempt + 1}): "
                f"status={exec_status} | verify_ok={verify_rep['ok']}"
            )

            # ------------------------------------------------------------------
            # STEP 2.5: DIAGNOSE PHASE
            # ------------------------------------------------------------------
            csv_preview = "\n".join(smart_read_file(df, 1, max_tail_lines) for df in data_files[:2]) or "No CSV files."
            diag_prompt = hm.load_node_prompt(
                "solver_reporter",
                {
                    "node_phase": "diagnose",
                    "task_id": task_id,
                    "execution_status": exec_status,
                    "stderr_traceback": stderr_traceback,
                    "log_tail": log_tail,
                    "intermediate_results_digest": json.dumps(results_digest, indent=2),
                    "csv_preview": csv_preview,
                    "verification_report": json.dumps(verify_rep, indent=2),
                    "stagnation_detected": str(stagnation_detected),
                    "inner_retry_count": str(attempt),
                },
            )
            diag: DiagnoserOutput = llm_det.with_structured_output(DiagnoserOutput).invoke(diag_prompt)

            stagnation_note = (
                " [STAGNATION WARNING: Attempt emitted identical code SHA-256 to prior attempt; you MUST modify the implementation]"
                if stagnation_detected
                else ""
            )
            diagnostic_feedback = (diag.feedback_for_next_node or diag.diagnosis) + stagnation_note

            inner_loop_trace.append({
                "attempt": attempt + 1,
                "ran_derive": should_derive,
                "code_sha256": new_code_hash,
                "stagnation_detected": stagnation_detected,
                "exec_status": exec_status,
                "verify_ok": verify_rep.get("ok", False),
                "npz_summary": results_digest.get("arrays", {}),
                "issues": verify_rep.get("issues", []),
                "diag_status": diag.status,
                "diagnosis": diagnostic_feedback,
            })

            if not (exec_status == "success" and verify_rep.get("ok") and diag.status == "converged_and_verified"):
                if diag.status == "strategic_error" or (attempt + 1 >= max_inner_retries):
                    outer = state.get("outer_loop_count", 0)
                    if outer < max_outer_loops:
                        is_oom = "MemoryError" in stderr_traceback or "OutOfMemory" in log_tail or diag.status == "strategic_error"
                        planner_entry = "requirement" if is_oom else "plan"
                        logger.warning(
                            f"[{task_id}] Solver escalating to Planner (outer={outer + 1}, entry={planner_entry})."
                        )
                        return {
                            "derivation_path": derivation_path,
                            "derivation_summary": derivation_summary,
                            "generated_code_path": script_path,
                            "execution_status": exec_status,
                            "log_file_path": rel_log_path,
                            "log_tail": log_tail,
                            "stderr_traceback": stderr_traceback,
                            "verification_report": verify_rep,
                            "intermediate_results_digest": results_digest,
                            "generated_results": results_files,
                            "generated_plots": plots,
                            "generated_data": data_files,
                            "inner_retry_count": attempt + 1,
                            "inner_loop_trace": inner_loop_trace,
                            "outer_loop_count": outer + 1,
                            "planner_entry_phase": planner_entry,
                            "diagnostic_feedback": diagnostic_feedback,
                            "next_node": "planner",
                        }
                    logger.warning(
                        f"[{task_id}] Circuit breaker reached (outer={outer}); proceeding to Step 2.6 diagnostic report."
                    )
                else:
                    attempt += 1
                    if diag.status == "derivation_error" or "NaN" in str(verify_rep.get("issues", "")):
                        logger.warning(f"[{task_id}] Stencil/NaN error detected -> looping to Step 2.1 (Derive).")
                        should_derive = True
                    else:
                        logger.info(f"[{task_id}] Code/Plot error detected -> looping to Step 2.3 (Code & Debug).")
                        should_derive = False
                    continue

        # ------------------------------------------------------------------
        # STEP 2.6: REPORT SYNTHESIS PHASE
        # ------------------------------------------------------------------
        missing_artifacts = [
            rel_req
            for rel_req in (
                spec.get("required_result_files", [])
                + spec.get("required_data_files", [])
                + spec.get("required_plots", [])
            )
            if not (active_ws / rel_req).exists()
        ]
        if missing_artifacts and attempt + 1 < max_inner_retries:
            diagnostic_feedback = f"Step 2.6 Cross-Check: Missing spec deliverables {missing_artifacts}; regenerate in solver.py."
            logger.warning(f"[{task_id}] {diagnostic_feedback} -> looping to Step 2.3.")
            skip_to_report = False
            should_derive = False
            attempt += 1
            continue

        data_contents = "\n\n".join(smart_read_file(df, 1, max_tail_lines) for df in data_files)
        deriv_excerpt = (
            smart_read_file(derivation_path, 1, max_tail_lines)
            if derivation_path and (active_ws / derivation_path).exists()
            else "None"
        )

        report_prompt = hm.load_node_prompt(
            "solver_reporter",
            {
                "node_phase": "report",
                "task_id": task_id,
                "simulation_spec": json.dumps(spec, indent=2),
                "derivation_summary": deriv_excerpt,
                "generated_solver_script": script_path,
                "saved_simulation_results": str(results_files),
                "intermediate_results_digest": json.dumps(results_digest, indent=2),
                "generated_plots": str(plots),
                "quantitative_csv_metrics": data_contents,
                "execution_log_tail": log_tail,
                "judge_feedback_if_revising_report": judge_feedback,
            },
        )
        try:
            rep_out: ReporterOutput = llm_gen.with_structured_output(ReporterOutput).invoke(report_prompt)
            report_md = _ensure_str(rep_out.report_markdown)
        except Exception:
            raw_resp = llm_gen.invoke(report_prompt)
            report_md = _ensure_str(getattr(raw_resp, "content", raw_resp))

        write_res = smart_write_file(rel_report_path, report_md, author_node="solver_reporter")
        report_disk_file = active_ws / rel_report_path
        logger.info(
            f"[{task_id}] Solver & Reporter (Step 2.6 Report): {write_res} "
            f"(verified_bytes={report_disk_file.stat().st_size if report_disk_file.exists() else 0})"
        )

        return {
            "derivation_path": derivation_path,
            "derivation_summary": derivation_summary,
            "derivation_audit_notes": derivation_audit_notes,
            "generated_code_path": script_path,
            "required_packages": required_packages,
            "execution_status": exec_status,
            "log_file_path": rel_log_path,
            "log_tail": log_tail,
            "stderr_traceback": stderr_traceback,
            "verification_report": verify_rep,
            "intermediate_results_digest": results_digest,
            "generated_results": results_files,
            "generated_plots": plots,
            "generated_data": data_files,
            "report_path": rel_report_path,
            "final_report_markdown": report_md,
            "inner_retry_count": attempt,
            "inner_loop_trace": inner_loop_trace,
            "diagnostic_feedback": "Converged, verified, and final report synthesized.",
            "next_node": "judge_reflector",
        }

    return {"next_node": "judge_reflector"}