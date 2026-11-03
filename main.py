# main.py
import argparse
import sys
from pathlib import Path
from typing import List, Tuple
from core import PROJECT_ROOT, load_config
from core.logger import setup_logger
from core.security import set_active_workspace, TASKS_DIR
from core.tools_io import smart_write_file
from agent.graph import build_autosci_graph

logger = setup_logger("Main", mode="system")


def discover_tasks(cli_tasks: List[str] | None = None) -> List[Tuple[str, Path]]:
    """
    Discovers tasks sequentially from `tasks/<task_id>/<task_id>.md`
    (or standalone `tasks/<task_id>.md` files).
    """
    discovered: List[Tuple[str, Path]] = []

    if cli_tasks:
        for raw in cli_tasks:
            p = (Path(raw) if Path(raw).is_absolute() else (PROJECT_ROOT / raw)).resolve()
            if p.is_dir():
                exact = p / f"{p.name}.md"
                md_candidates = [exact] if exact.exists() else sorted(p.glob("*.md"))
                if md_candidates:
                    discovered.append((p.name, md_candidates[0].resolve()))
            elif p.is_file() and p.suffix.lower() == ".md":
                task_id = p.parent.name if p.parent != TASKS_DIR else p.stem
                discovered.append((task_id, p.resolve()))
        return discovered

    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    for item in sorted(TASKS_DIR.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            exact = item / f"{item.name}.md"
            md_files = [exact] if exact.exists() else sorted(item.glob("*.md"))
            if md_files:
                discovered.append((item.name, md_files[0].resolve()))
        elif item.is_file() and item.suffix.lower() == ".md":
            discovered.append((item.stem, item.resolve()))

    return discovered


def main():
    parser = argparse.ArgumentParser(description="AutoSciAgent 3-Node Sequential 1D/2D/3D PDE Runner")
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Optional task paths (e.g., tasks/<task_id>/<task_id>.md) to run sequentially.",
    )
    args = parser.parse_args()

    cfg = load_config()
    agent_cfg = cfg["agent"]
    paths_cfg = cfg["paths"]
    recursion_limit = int(agent_cfg["recursion_limit"])
    default_report_rel_path = paths_cfg["report_relative_path"]

    task_queue = discover_tasks(args.tasks)
    if not task_queue:
        logger.error(f"No task markdown files found in '{TASKS_DIR}'.")
        sys.exit(1)

    logger.info(
        f"Initializing AutoSciAgent [v{cfg.get('version')}] "
        f"| Model: {cfg['llm']['model_id']} "
        f"| Task Queue ({len(task_queue)}): {[tid for tid, _ in task_queue]}"
    )

    app = build_autosci_graph()

    for idx, (task_id, task_md_path) in enumerate(task_queue, start=1):
        active_ws = set_active_workspace(task_id)
        rel_task_md = str(task_md_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        logger.info("=" * 72)
        logger.info(
            f"STARTING TASK [{idx}/{len(task_queue)}]: {task_id} "
            f"| Input: {rel_task_md} | Workspace: workspace/{task_id}/"
        )
        logger.info("=" * 72)

        initial_state = {
            "task_id": task_id,
            "task_file_path": rel_task_md,
            "planner_entry_phase": "requirement",
            "req_retry_count": 0,
            "inner_retry_count": 0,
            "outer_loop_count": 0,
            "rolling_summary": f"Initialized isolated workspace/{task_id}/.",
        }

        final_state = dict(initial_state)
        step_idx = 0
        for event in app.stream(initial_state, {"recursion_limit": recursion_limit}):
            for node_name, state_update in event.items():
                step_idx += 1
                logger.info(
                    f"[{task_id}] Step {step_idx:02d} | Completed Node: [{node_name.upper()}] "
                    f"-> Next: [{state_update.get('next_node')}]"
                )
                final_state.update(state_update)

        report_rel_path = final_state.get("report_path", default_report_rel_path)
        report_disk_path = active_ws / report_rel_path
        if report_disk_path.exists() and report_disk_path.stat().st_size > 0:
            logger.info(
                f"[{task_id}] Verified final report on disk: workspace/{active_ws.name}/{report_rel_path} "
                f"({report_disk_path.stat().st_size} bytes)"
            )
        elif final_state.get("final_report_markdown"):
            save_status = smart_write_file(
                report_rel_path,
                final_state["final_report_markdown"],
                author_node="main_fallback",
            )
            logger.info(f"[{task_id}] Fallback report save: {save_status}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by operator.")
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Unhandled system error: {exc}")
        sys.exit(1)