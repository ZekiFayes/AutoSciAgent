# core/logger.py
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_base_log_dir() -> Path:
    try:
        if CONFIG_PATH.exists():
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            rel_path = cfg.get("paths", {}).get("system_logs_dir", "logs")
            return (PROJECT_ROOT / rel_path).resolve()
    except Exception:
        pass
    return (PROJECT_ROOT / "logs").resolve()


def setup_logger(name: str, mode: str = "system") -> logging.Logger:
    base_dir = _get_base_log_dir()
    log_dir = base_dir / mode
    log_file = log_dir / f"run_{_RUN_TIMESTAMP}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger_key = f"AutoSciAgent.{mode}.{name}"
    logger = logging.getLogger(logger_key)
    logger.disabled = False
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger