# core/security.py
from pathlib import Path
from typing import Literal, Tuple
from core import PROJECT_ROOT, load_config
from core.logger import setup_logger

logger = setup_logger("Security-Jail", mode="system")

_cfg = load_config()
TASKS_DIR = (PROJECT_ROOT / _cfg["paths"]["tasks_dir"]).resolve()
WORKSPACE_ROOT = (PROJECT_ROOT / _cfg["paths"]["workspace_root"]).resolve()

_ACTIVE_TASK_WORKSPACE: Path = (WORKSPACE_ROOT / "task_01").resolve()


def set_active_workspace(task_id: str) -> Path:
    global _ACTIVE_TASK_WORKSPACE
    ws_dir = (WORKSPACE_ROOT / task_id).resolve()

    for subdir in (
        ".versions",
        ".trash",
        "specs",
        "derivations",
        "scripts",
        "outputs/run_logs",
        "outputs/results",
        "outputs/data",
        "outputs/plots",
        "outputs/reports",
    ):
        (ws_dir / subdir).mkdir(parents=True, exist_ok=True)

    _ACTIVE_TASK_WORKSPACE = ws_dir
    logger.info(f"Activated isolated task workspace: workspace/{task_id}/")
    return _ACTIVE_TASK_WORKSPACE


def get_active_workspace() -> Path:
    _ACTIVE_TASK_WORKSPACE.mkdir(parents=True, exist_ok=True)
    return _ACTIVE_TASK_WORKSPACE


def classify_and_resolve_path(
    target_path: str | Path,
    allow_outside_workspace: bool = False,
    write_operation: bool = False,
) -> Tuple[Path, Literal["INSIDE_WORKSPACE", "OUTSIDE_WORKSPACE"]]:
    active_ws = get_active_workspace()
    raw = Path(target_path)

    if not raw.is_absolute():
        parts = raw.parts
        if parts and parts[0].lower() in ("tasks", "harness", "library", "configs", "logs", ".venv"):
            resolved = (PROJECT_ROOT / raw).resolve()
        elif parts and parts[0].lower() == "workspace":
            if len(parts) > 1 and parts[1] == active_ws.name:
                resolved = (PROJECT_ROOT / raw).resolve()
            else:
                resolved = (active_ws / Path(*parts[1:])).resolve()
        else:
            resolved = (active_ws / raw).resolve()
    else:
        resolved = raw.resolve()

    if resolved.is_relative_to(TASKS_DIR):
        if write_operation:
            raise PermissionError(f"Security Violation: Decoupled input folder '{resolved}' is strictly read-only.")
        return resolved, "OUTSIDE_WORKSPACE"

    if resolved.is_relative_to(active_ws):
        if write_operation and resolved == active_ws:
            raise PermissionError("Security Violation: Cannot overwrite root workspace directory.")
        return resolved, "INSIDE_WORKSPACE"

    if not allow_outside_workspace:
        logger.error(f"Blocked unauthorized outside-workspace access ({active_ws.name}): {resolved}")
        raise PermissionError(
            f"Security Violation: Path '{resolved}' is outside active workspace 'workspace/{active_ws.name}/'."
        )

    return resolved, "OUTSIDE_WORKSPACE"