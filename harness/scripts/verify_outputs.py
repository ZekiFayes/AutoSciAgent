# harness/scripts/verify_outputs.py
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np


def _load_min_png_bytes() -> int:
    cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return int(cfg.get("agent", {}).get("min_valid_png_bytes", 500))
        except Exception:
            pass
    return 500


def verify_workspace_outputs(workspace_dir: Path, min_png_bytes: int | None = None) -> dict:
    """
    Spec-aware output verifier:
    1. Reads `specs/simulation_spec.json` (if present) to verify all required `.npz`, `.csv`, and `.png` paths
       and ensure all comparison scenarios are represented.
    2. Reopens `.npz`/`.npy` arrays and `.csv` cells to guarantee zero `NaN`/`Inf` values.
    """
    min_bytes = min_png_bytes if min_png_bytes is not None else _load_min_png_bytes()
    png_signature = b"\x89PNG\r\n\x1a\n"

    spec_path = workspace_dir / "specs" / "simulation_spec.json"
    expected_scenarios = 1
    req_results: list[str] = []
    req_data: list[str] = []
    req_plots: list[str] = []
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            expected_scenarios = max(1, len(spec.get("comparison_scenarios", [])))
            req_results = spec.get("required_result_files", [])
            req_data = spec.get("required_data_files", [])
            req_plots = spec.get("required_plots", [])
        except Exception:
            pass

    results_dir = workspace_dir / "outputs" / "results"
    data_dir = workspace_dir / "outputs" / "data"
    plots_dir = workspace_dir / "outputs" / "plots"

    issues = []
    verified_results = []
    verified_csvs = []
    verified_pngs = []

    for rel_req in req_results + req_data + req_plots:
        if not (workspace_dir / rel_req).exists():
            issues.append(f"Spec-required deliverable '{rel_req}' is missing on disk.")

    res_files = (
        sorted(list(results_dir.glob("*.npz")) + list(results_dir.glob("*.npy")))
        if results_dir.exists()
        else []
    )
    if not res_files:
        issues.append("No saved simulation field files (.npz or .npy) found in outputs/results/.")
    else:
        for rf in res_files:
            try:
                loaded = np.load(rf)
                if rf.suffix == ".npz":
                    keys = list(loaded.files)
                    if len(keys) < expected_scenarios:
                        issues.append(
                            f"NPZ '{rf.name}' has {len(keys)} arrays, expected at least {expected_scenarios} scenarios."
                        )
                    for k in keys:
                        arr = np.asarray(loaded[k])
                        if arr.size == 0 or not np.all(np.isfinite(arr)):
                            issues.append(f"Array '{k}' in '{rf.name}' is empty or contains NaN/Inf.")
                else:
                    arr = np.asarray(loaded)
                    if arr.size == 0 or not np.all(np.isfinite(arr)):
                        issues.append(f"NPY '{rf.name}' is empty or contains NaN/Inf.")
                verified_results.append(rf.name)
            except Exception as e:
                issues.append(f"Failed to load simulation result '{rf.name}': {e}")

    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        issues.append("No CSV metric files found in outputs/data/.")
    else:
        for csv_file in sorted(data_dir.glob("*.csv")):
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if len(rows) < expected_scenarios + 1:
                    issues.append(
                        f"CSV '{csv_file.name}' has {len(rows)} rows, expected header + {expected_scenarios} scenario rows."
                    )
                for r_idx, row in enumerate(rows[1:], start=2):
                    for cell in row:
                        try:
                            val = float(cell)
                            if math.isnan(val) or math.isinf(val):
                                issues.append(f"NaN/Inf detected in '{csv_file.name}' at row {r_idx}.")
                        except ValueError:
                            pass
                verified_csvs.append(csv_file.name)
            except Exception as e:
                issues.append(f"Failed to read CSV '{csv_file.name}': {e}")

    if not plots_dir.exists() or not list(plots_dir.glob("*.png")):
        issues.append("No PNG plot files found in outputs/plots/.")
    else:
        for png_file in sorted(plots_dir.glob("*.png")):
            size = png_file.stat().st_size
            header = png_file.read_bytes()[: len(png_signature)]
            if size < min_bytes or header != png_signature:
                issues.append(f"Plot '{png_file.name}' is corrupted or empty ({size} bytes).")
            else:
                verified_pngs.append(png_file.name)

    ok = (
        len(issues) == 0
        and len(verified_results) > 0
        and len(verified_csvs) > 0
        and len(verified_pngs) > 0
    )
    return {
        "ok": ok,
        "verified_results": verified_results,
        "verified_csvs": verified_csvs,
        "verified_pngs": verified_pngs,
        "issues": issues,
    }


if __name__ == "__main__":
    ws = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    report = verify_workspace_outputs(ws)
    print(json.dumps(report))
    sys.exit(0 if report["ok"] else 1)