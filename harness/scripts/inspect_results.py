# harness/scripts/inspect_results.py
import csv
import json
import sys
from pathlib import Path
import numpy as np


def _summarize_array(arr: np.ndarray) -> dict:
    arr_np = np.asarray(arr)
    nan_count = int(np.isnan(arr_np).sum()) if np.issubdtype(arr_np.dtype, np.number) else 0
    inf_count = int(np.isinf(arr_np).sum()) if np.issubdtype(arr_np.dtype, np.number) else 0
    finite_vals = arr_np[np.isfinite(arr_np)] if np.issubdtype(arr_np.dtype, np.number) else np.array([])

    summary = {
        "shape": list(arr_np.shape),
        "dtype": str(arr_np.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": round(float(finite_vals.min()), 4) if finite_vals.size > 0 else None,
        "max": round(float(finite_vals.max()), 4) if finite_vals.size > 0 else None,
        "mean": round(float(finite_vals.mean()), 4) if finite_vals.size > 0 else None,
    }
    if finite_vals.size > 0 and arr_np.ndim >= 1:
        summary["boundary_first_mean"] = round(float(np.nanmean(arr_np[0])), 4)
        summary["boundary_last_mean"] = round(float(np.nanmean(arr_np[-1])), 4)
        mid_idx = tuple(s // 2 for s in arr_np.shape)
        summary["center_val"] = round(float(arr_np[mid_idx]), 4)
    return summary


def inspect_workspace_results(workspace_dir: Path, target_rel_path: str | None = None) -> dict:
    """
    Builds a structured numerical digest of `.npz`/`.npy` field arrays and `.csv` tables
    in `outputs/results/` (or falls back to the latest archived `.versions/outputs/results/`
    if the most recent run crashed before writing outputs) so the LLM can debug in Step 2.3.
    """
    digest: dict = {"source": "outputs", "arrays": {}, "csv_rows": []}

    if target_rel_path:
        cand = (workspace_dir / target_rel_path).resolve()
        res_files = [cand] if cand.exists() else []
    else:
        results_dir = workspace_dir / "outputs" / "results"
        res_files = (
            sorted(list(results_dir.glob("*.npz")) + list(results_dir.glob("*.npy")))
            if results_dir.exists()
            else []
        )
        if not res_files:
            ver_res_dir = workspace_dir / ".versions" / "outputs" / "results"
            if ver_res_dir.exists():
                archived = sorted(
                    list(ver_res_dir.glob("*.npz")) + list(ver_res_dir.glob("*.npy")),
                    key=lambda p: p.stat().st_mtime,
                )
                if archived:
                    res_files = [archived[-1]]
                    digest["source"] = f".versions/outputs/results/{archived[-1].name} (fallback from prior run)"

    for rf in res_files:
        try:
            loaded = np.load(rf)
            if rf.suffix == ".npz":
                for k in loaded.files:
                    digest["arrays"][f"{rf.name}:{k}"] = _summarize_array(loaded[k])
            else:
                digest["arrays"][rf.name] = _summarize_array(loaded)
        except Exception as e:
            digest["arrays"][rf.name] = {"error": str(e)}

    data_dir = workspace_dir / "outputs" / "data"
    csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    if not csv_files:
        ver_data_dir = workspace_dir / ".versions" / "outputs" / "data"
        if ver_data_dir.exists():
            archived_csvs = sorted(ver_data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            if archived_csvs:
                csv_files = [archived_csvs[-1]]

    for cf in csv_files[:2]:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                digest["csv_rows"] = list(csv.reader(f))[:15]
        except Exception as e:
            digest["csv_rows"] = [[f"Error reading CSV: {e}"]]

    return digest


if __name__ == "__main__":
    ws = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    rel = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(inspect_workspace_results(ws, rel), indent=2))