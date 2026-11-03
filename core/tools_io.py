# core/tools_io.py
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
from core import load_config
from core.security import classify_and_resolve_path, get_active_workspace
from core.tools_hitl import request_human_approval
from core.logger import setup_logger

logger = setup_logger("IO-Tools", mode="system")


def _archive_existing_file_cow(existing_path: Path, author_node: str = "agent", move_file: bool = False) -> Path:
    """
    Archives existing workspace files into `workspace/<task_id>/.versions/`:
    - `move_file=False` (default for `smart_write_file`): copies the prior revision before writing the new one.
    - `move_file=True` (used by `archive_and_clear_outputs_cow` before solver execution): moves prior output
      files out of `outputs/{results,data,plots}` into `.versions/` so stale files never cause false-positives,
      while keeping them accessible for fallback debug inspection.
    """
    active_ws = get_active_workspace()
    versions_dir = active_ws / ".versions"
    manifest_path = versions_dir / "manifest.json"
    versions_dir.mkdir(parents=True, exist_ok=True)

    rel_str = str(existing_path.relative_to(active_ws)).replace("\\", "/")

    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    history = manifest.get(rel_str, [])
    version_num = len(history) + 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_rel = Path(rel_str).with_name(f"{existing_path.stem}.v{version_num:03d}_{ts}{existing_path.suffix}")
    archive_target = versions_dir / archive_rel
    archive_target.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(existing_path.read_bytes()).hexdigest()[:12]
    if move_file:
        shutil.move(str(existing_path), str(archive_target))
    else:
        shutil.copy2(existing_path, archive_target)

    history.append({
        "version": version_num,
        "archived_at": ts,
        "author_node": author_node,
        "sha256_prefix": digest,
        "archive_path": str(archive_target.relative_to(active_ws)).replace("\\", "/"),
    })
    manifest[rel_str] = history
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"[{active_ws.name}] CoW Archive: saved v{version_num:03d} of '{rel_str}' -> {archive_target.name}")
    return archive_target


def archive_and_clear_outputs_cow(author_node: str = "solver_pre_exec") -> None:
    """
    Clean-Slate Move Guard for Step 2.4:
    Moves (`shutil.move`) all existing files in `outputs/{results,data,plots}` to `.versions/`
    right before launching `solver.py` so stale files from an earlier attempt never pollute verification.
    """
    active_ws = get_active_workspace()
    for subdir in ("outputs/results", "outputs/data", "outputs/plots"):
        dpath = active_ws / subdir
        if dpath.exists():
            for existing_f in sorted(dpath.iterdir()):
                if existing_f.is_file():
                    _archive_existing_file_cow(existing_f, author_node=author_node, move_file=True)


def smart_read_file(path_str: str, start_line: int = 1, max_lines: int | None = None, allow_outside: bool = False) -> str:
    try:
        cfg = load_config()
        line_limit = max_lines if max_lines is not None else int(cfg["agent"]["max_read_lines"])

        safe_path, _ = classify_and_resolve_path(path_str, allow_outside_workspace=allow_outside, write_operation=False)
        if not safe_path.exists():
            return f"Error: File '{path_str}' does not exist."
        if safe_path.is_dir():
            return f"Error: '{path_str}' is a directory."

        lines = safe_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total, start_idx + line_limit)
        snippet = "\n".join(f"{i+1:4d}: {lines[i]}" for i in range(start_idx, end_idx))

        header = f"[File: {path_str} | Lines {start_idx+1}-{end_idx} of {total}]\n"
        footer = f"\n... [{total - end_idx} lines omitted to prevent OOM.]" if end_idx < total else ""
        return header + snippet + footer
    except Exception as e:
        return f"I/O Read Error: {str(e)}"


def smart_write_file(
    path_str: str,
    content: str | list,
    author_node: str = "agent",
    allow_outside: bool = False,
) -> str:
    try:
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        elif not isinstance(content, str):
            content = str(content)

        active_ws = get_active_workspace()
        safe_path, zone = classify_and_resolve_path(
            path_str, allow_outside_workspace=allow_outside, write_operation=True
        )

        if zone == "OUTSIDE_WORKSPACE":
            if not request_human_approval("OUTSIDE-WORKSPACE WRITE", f"Write/modify '{safe_path}' ({len(content)} bytes)"):
                return "Operation aborted by human operator (Outside-workspace write rejected)."
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding="utf-8")
            return f"Success: Wrote {len(content)} bytes to outside-workspace path '{safe_path.name}'."

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        archive_note = " (New file)"
        if safe_path.exists():
            archived = _archive_existing_file_cow(safe_path, author_node=author_node, move_file=False)
            archive_note = f" (Prior version preserved at .versions/.../{archived.name})"

        safe_path.write_text(content, encoding="utf-8")
        rel = str(safe_path.relative_to(active_ws)).replace("\\", "/")
        return f"Success: Wrote {len(content)} bytes to workspace/{active_ws.name}/{rel}{archive_note}."
    except Exception as e:
        logger.error(f"I/O Write Error on '{path_str}': {e}")
        return f"I/O Write Error: {str(e)}"


def smart_remove_item(path_str: str, allow_outside: bool = False) -> str:
    try:
        active_ws = get_active_workspace()
        trash_dir = active_ws / ".trash"
        safe_path, zone = classify_and_resolve_path(
            path_str, allow_outside_workspace=allow_outside, write_operation=True
        )
        if not safe_path.exists():
            return f"Path '{path_str}' does not exist."

        if zone == "OUTSIDE_WORKSPACE":
            if not request_human_approval("OUTSIDE-WORKSPACE REMOVE", f"Delete '{safe_path}'"):
                return "Operation aborted by human operator."
            if safe_path.is_dir():
                shutil.rmtree(safe_path)
            else:
                safe_path.unlink()
            return f"Success: Deleted outside-workspace item '{safe_path.name}'."

        trash_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_target = trash_dir / f"{ts}_{safe_path.name}"
        shutil.move(str(safe_path), str(trash_target))
        return f"Success: Moved workspace/{active_ws.name}/{safe_path.relative_to(active_ws)} to .trash/{trash_target.name}."
    except Exception as e:
        return f"I/O Remove Error: {str(e)}"


@tool
def mcp_read_file(file_path: str, start_line: int = 1, max_lines: int = 150) -> str:
    """MCP Debug Tool: Read a bounded line window from a log, CSV, or script in the active task workspace."""
    return smart_read_file(file_path, start_line=start_line, max_lines=max_lines)


@tool
def mcp_write_file(file_path: str, content: str) -> str:
    """MCP Tool: Write or revise a file in the active task workspace (Copy-on-Write non-overwriting)."""
    return smart_write_file(file_path, content, author_node="mcp_tool")


@tool
def mcp_remove_item(target_path: str) -> str:
    """MCP Tool: Soft-delete a file/folder in the active task workspace to .trash/ or outside workspace with HITL."""
    return smart_remove_item(target_path)