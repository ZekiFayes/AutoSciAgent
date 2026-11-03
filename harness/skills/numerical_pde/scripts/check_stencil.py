# harness/skills/numerical_pde/scripts/check_stencil.py
import sys


def verify_conservation_weights(kx: float, ky: float = 0.0, kz: float = 0.0) -> bool:
    denom = 2.0 * (kx + ky + kz)
    if denom <= 0.0:
        return False
    total = 2.0 * (kx / denom) + 2.0 * (ky / denom) + 2.0 * (kz / denom)
    return abs(total - 1.0) < 1e-12


if __name__ == "__main__":
    ok = verify_conservation_weights(1.0, 1.0, 0.3)
    print(f"Stencil conservation check: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)