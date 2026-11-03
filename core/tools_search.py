# core/tools_search.py
import json
import re
from langchain_core.tools import tool
from core import load_config
from core.security import classify_and_resolve_path, get_active_workspace
from harness.scripts.inspect_results import inspect_workspace_results


def grep_workspace(pattern: str, subdir: str = ".") -> str:
    try:
        cfg = load_config()
        max_matches = int(cfg["agent"]["max_grep_matches"])

        active_ws = get_active_workspace()
        safe_dir, _ = classify_and_resolve_path(subdir, allow_outside_workspace=False, write_operation=False)
        regex = re.compile(pattern, re.IGNORECASE)
        matches = []
        for path in sorted(safe_dir.rglob("*")):
            if not path.is_file() or any(part in path.parts for part in (".versions", ".trash")):
                continue
            if path.suffix.lower() in (".png", ".pdf", ".npy", ".npz", ".pyc"):
                continue
            for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if regex.search(line):
                    rel = str(path.relative_to(active_ws)).replace("\\", "/")
                    matches.append(f"{rel}:{idx}: {line.strip()[:120]}")
                    if len(matches) >= max_matches:
                        return "\n".join(matches) + f"\n... [Truncated at {max_matches} matches]"
        return "\n".join(matches) if matches else f"No matches found for pattern '{pattern}'."
    except Exception as e:
        return f"Search Error: {str(e)}"


@tool
def mcp_search_workspace(regex_pattern: str, subdir: str = ".") -> str:
    """MCP Debug Tool: Search for a regex pattern across logs, CSVs, and scripts in the active task workspace."""
    return grep_workspace(regex_pattern, subdir=subdir)


@tool
def mcp_inspect_results(relative_path: str = "") -> str:
    """
    MCP Debug Tool: Inspect intermediate `.npz`/`.npy` field arrays and `.csv` tables
    (reporting shapes, min/max/mean, NaN/Inf counts, and boundary slices T[0], T[-1]).
    """
    active_ws = get_active_workspace()
    rel = relative_path.strip() or None
    return json.dumps(inspect_workspace_results(active_ws, rel), indent=2)