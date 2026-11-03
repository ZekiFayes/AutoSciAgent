# harness/scripts/hardware_probe.py
import json
import os
import platform
from pathlib import Path


def probe_hardware() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.json"
    hw_defaults = {
        "os": platform.system(),
        "cpu_cores": os.cpu_count() or 1,
        "gpu_available": False,
        "gpu_backend": "cpu_numpy",
        "device_name": platform.processor() or "CPU",
        "free_ram_mb": 4096,
    }
    if cfg_path.exists():
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8")).get("hardware_defaults", {})
            hw_defaults["free_ram_mb"] = int(loaded.get("free_ram_mb", hw_defaults["free_ram_mb"]))
        except Exception:
            pass

    bytes_per_mb = 1024 * 1024
    try:
        import cupy as cp  # type: ignore
        if cp.cuda.is_available():
            hw_defaults["gpu_available"] = True
            hw_defaults["gpu_backend"] = "cupy"
            free_b, _ = cp.cuda.runtime.memGetInfo()
            hw_defaults["free_ram_mb"] = int(free_b // bytes_per_mb)
            return hw_defaults
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            hw_defaults["gpu_available"] = True
            hw_defaults["gpu_backend"] = "torch_cuda"
            hw_defaults["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return hw_defaults


if __name__ == "__main__":
    print(json.dumps(probe_hardware()))