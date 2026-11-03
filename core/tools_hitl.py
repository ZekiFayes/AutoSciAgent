# core/tools_hitl.py
import os
from core import load_config
from core.logger import setup_logger

logger = setup_logger("HITL-Gate", mode="system")


def request_human_approval(action_type: str, details: str) -> bool:
    if os.getenv("AUTOSCI_AUTO_APPROVE", "0") == "1" or os.getenv("AUTOSCI_TEST_MODE", "0") == "1":
        logger.info(f"Auto-approved [{action_type}] (Test/Auto mode): {details}")
        return True

    cfg = load_config()
    if not cfg["agent"]["hitl_enabled"]:
        logger.info(f"HITL disabled in config. Auto-approving [{action_type}]: {details}")
        return True

    print("\n" + "=" * 72)
    print(f" [HUMAN-IN-THE-LOOP SECURITY GATE] Action: {action_type}")
    print("-" * 72)
    print(f" Details: {details}")
    print("=" * 72)

    while True:
        choice = input("Approve this operation? [Y/n]: ").strip().lower()
        if choice in ("y", "yes", ""):
            logger.info(f"Operator APPROVED [{action_type}]: {details}")
            return True
        if choice in ("n", "no"):
            logger.warning(f"Operator REJECTED [{action_type}]: {details}")
            return False
        print("Invalid input. Please enter 'Y' or 'n'.")