# harness/plugins/runtime_middleware.py
from core import load_config


def apply_post_execution_middleware(
    raw_log: str,
    retry_count: int,
    results_found: list,
    plots_found: list,
    data_found: list,
) -> str:
    cfg = load_config()
    max_line_chars = int(cfg["agent"]["middleware_max_line_chars"])

    processed_lines = []
    for line in raw_log.splitlines():
        if len(line) > max_line_chars:
            processed_lines.append(line[:max_line_chars] + " ... [LINE TRUNCATED BY C8 MIDDLEWARE]")
        else:
            processed_lines.append(line)

    output = "\n".join(processed_lines)

    if "MemoryError" in output or "OutOfMemory" in output:
        output += "\n[MIDDLEWARE HINT]: Memory exhaustion detected. Reduce 1D/2D/3D grid size or use vectorized updates."
    if "TclError" in output or "qt.qpa" in output:
        output += "\n[MIDDLEWARE HINT]: Headless display error. Call `matplotlib.use('Agg')` before `import matplotlib.pyplot as plt`."
    if not results_found or not plots_found or not data_found:
        output += (
            f"\n[WATCHDOG ALERT (Retry {retry_count})]: Deliverable check incomplete "
            f"(results={len(results_found)}, plots={len(plots_found)}, data_files={len(data_found)}). "
            "Ensure script saves `.npz` to `outputs/results/`, `.csv` to `outputs/data/`, and `.png` to `outputs/plots/`."
        )
    return output