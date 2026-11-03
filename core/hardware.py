# core/hardware.py
import json
import subprocess
import sys
from core import PROJECT_ROOT, load_config
from core.logger import setup_logger

logger = setup_logger("Hardware-Probe", mode="system")


def detect_hardware_profile() -> dict:
    cfg = load_config()
    timeout_sec = int(cfg["agent"]["hardware_probe_timeout_sec"])
    hw_defaults = dict(cfg["hardware_defaults"])

    probe_script = PROJECT_ROOT / cfg["paths"]["harness_dir"] / "scripts" / "hardware_probe.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(probe_script)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except Exception as e:
        logger.warning(f"Hardware probe fallback to config defaults: {e}")

    return hw_defaults


def estimate_grid_memory_mb(grid_spec: dict, spatial_dims: int = 2) -> float:
    cfg = load_config()
    hw_cfg = cfg["hardware_defaults"]
    bytes_per_float = int(hw_cfg["bytes_per_float"])

    nx = max(1, int(grid_spec.get("nx", 1)))
    ny = max(1, int(grid_spec.get("ny", 1))) if spatial_dims >= 2 else 1
    nz = max(1, int(grid_spec.get("nz", 1))) if spatial_dims == 3 else 1

    if spatial_dims == 1:
        num_arrays = int(hw_cfg["arrays_1d"])
    elif spatial_dims == 2:
        num_arrays = int(hw_cfg["arrays_2d"])
    else:
        num_arrays = int(hw_cfg["arrays_3d"])

    bytes_per_mb = 1024 * 1024
    total_bytes = nx * ny * nz * bytes_per_float * num_arrays
    return round(total_bytes / bytes_per_mb, 4)


def clamp_grid_to_memory_limit(grid_spec: dict, spatial_dims: int, max_allowed_mb: float) -> dict:
    """
    Deterministic fallback for Step 1.2 when Pre-Audit retries are exhausted:
    scales down `nx`, `ny`, `nz` proportionally until estimated memory fits within `max_allowed_mb`.
    """
    cfg = load_config()
    min_dim = int(cfg["hardware_defaults"].get("min_clamped_grid_dim", 8))
    clamped = dict(grid_spec)
    while estimate_grid_memory_mb(clamped, spatial_dims) > max_allowed_mb:
        changed = False
        for key in ("nx", "ny", "nz"):
            if key in clamped and int(clamped[key]) > min_dim:
                clamped[key] = max(min_dim, int(int(clamped[key]) * 0.75))
                changed = True
        if not changed:
            break
    return clamped