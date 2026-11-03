import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = bool(cp.cuda.is_available())
except Exception:
    cp = None
    GPU_AVAILABLE = False

xp = cp if GPU_AVAILABLE else np


def to_cpu(arr):
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def load_simulation_variables(spec_file_path: str = "specs/simulation_spec.json") -> dict:
    """Loads all physical, numerical, and output variables from specs/simulation_spec.json."""
    candidates = [
        Path(spec_file_path),
        Path(__file__).resolve().parent.parent / spec_file_path,
    ]
    for cand in candidates:
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Simulation specification not found at '{spec_file_path}'.")


def run_1d_simulation(spec_vars: dict, res_path: Path, csv_path: Path, plot_path: Path) -> None:
    grid_vars = spec_vars["grid_and_domain"]
    scenarios = spec_vars["comparison_scenarios"]
    plot_vars = spec_vars.get("plot_settings", {})

    nx = int(grid_vars["nx"])
    lx = float(grid_vars["lx"])
    t_end = float(grid_vars["t_end"])
    dt_requested = float(grid_vars["dt"])
    fo_max = float(grid_vars["fo_max"])
    num_snapshots = int(grid_vars["num_snapshots"])

    dx = lx / float(nx - 1)
    x_coords = np.linspace(0.0, lx, nx)
    saved_fields = {"x_coords": x_coords}
    final_profiles = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "max_temp_c", "min_temp_c", "mid_rod_temp_c", "time_steps", "fourier_number"])

        for sc in scenarios:
            case_name = str(sc["name"])
            alpha = float(sc["alpha"])
            t_init = float(sc["t_init"])
            t_left = float(sc["t_left"])
            t_right = float(sc["t_right"])

            dt_stable = fo_max * (dx * dx) / max(alpha, 1e-15)
            dt = min(dt_requested, dt_stable)
            n_steps = max(1, int(math.ceil(t_end / dt)))
            dt = t_end / float(n_steps)
            fo = alpha * dt / (dx * dx)

            T = xp.full(nx, t_init, dtype=xp.float64)
            T[0] = t_left
            T[-1] = t_right

            snap_indices = set(np.linspace(0, n_steps, num_snapshots, dtype=int).tolist())
            snapshots = [to_cpu(T).copy()]

            for step in range(1, n_steps + 1):
                T_old = T.copy()
                T[1:-1] = T_old[1:-1] + fo * (T_old[2:] - 2.0 * T_old[1:-1] + T_old[:-2])
                T[0] = t_left
                T[-1] = t_right
                if step in snap_indices:
                    snapshots.append(to_cpu(T).copy())

            T_final = to_cpu(T)
            spacetime_mat = np.vstack(snapshots)
            saved_fields[f"{case_name}_final"] = T_final
            saved_fields[f"{case_name}_spacetime"] = spacetime_mat
            final_profiles[case_name] = T_final

            mid_temp = float(T_final[nx // 2])
            writer.writerow([
                case_name,
                f"{float(T_final.max()):.3f}",
                f"{float(T_final.min()):.3f}",
                f"{mid_temp:.3f}",
                n_steps,
                f"{fo:.4f}",
            ])
            f.flush()
            np.savez_compressed(res_path, **saved_fields)

    fig_w = float(plot_vars.get("figsize_width", 11))
    fig_h = float(plot_vars.get("figsize_height", 8))
    dpi = int(plot_vars.get("dpi", 140))
    cmap_name = str(plot_vars.get("colormap", "inferno"))
    leg_fs = int(plot_vars.get("legend_fontsize", 8))

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    for idx, sc in enumerate(scenarios):
        case_name = str(sc["name"])
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(
            saved_fields[f"{case_name}_spacetime"],
            origin="lower",
            aspect="auto",
            extent=[0.0, lx, 0.0, t_end],
            cmap=cmap_name,
        )
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax_prof = axes[1, 1]
    for case_name, prof in final_profiles.items():
        ax_prof.plot(x_coords, prof, label=case_name)
    ax_prof.set_title(f"Final Temperature Profile T(x, t={t_end}s)")
    ax_prof.set_xlabel("x (m)")
    ax_prof.set_ylabel("Temperature (C)")
    ax_prof.legend(fontsize=leg_fs)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def run_2d_simulation(spec_vars: dict, res_path: Path, csv_path: Path, plot_path: Path) -> None:
    grid_vars = spec_vars["grid_and_domain"]
    scenarios = spec_vars["comparison_scenarios"]
    plot_vars = spec_vars.get("plot_settings", {})

    nx = int(grid_vars["nx"])
    ny = int(grid_vars["ny"])
    lx = float(grid_vars["lx"])
    ly = float(grid_vars["ly"])
    tol = float(grid_vars["tolerance"])
    max_iter = int(grid_vars["max_iter"])

    results = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "max_temp_c", "min_temp_c", "center_temp_c", "iterations", "residual"])

        for sc in scenarios:
            case_name = str(sc["name"])
            t_top = float(sc["t_top"])
            t_other = float(sc["t_other"])
            kx = float(sc["kx"])
            ky = float(sc["ky"])

            T = xp.full((ny, nx), t_other, dtype=xp.float64)
            T[-1, :] = t_top
            T[0, :] = t_other
            T[:, 0] = t_other
            T[:, -1] = t_other

            denom = 2.0 * (kx + ky)
            residual = 1.0
            it = 0
            for it in range(1, max_iter + 1):
                T_old = T.copy()
                T[1:-1, 1:-1] = (
                    kx * (T_old[1:-1, 2:] + T_old[1:-1, :-2])
                    + ky * (T_old[2:, 1:-1] + T_old[:-2, 1:-1])
                ) / denom
                residual = float(xp.max(xp.abs(T - T_old)))
                if residual < tol:
                    break

            T_cpu = to_cpu(T)
            results[case_name] = T_cpu
            center_temp = float(T_cpu[ny // 2, nx // 2])
            writer.writerow([
                case_name,
                f"{float(T_cpu.max()):.3f}",
                f"{float(T_cpu.min()):.3f}",
                f"{center_temp:.3f}",
                it,
                f"{residual:.2e}",
            ])
            f.flush()
            np.savez_compressed(res_path, **results)

    fig_w = float(plot_vars.get("figsize_width", 11))
    fig_h = float(plot_vars.get("figsize_height", 9))
    dpi = int(plot_vars.get("dpi", 140))
    cmap_name = str(plot_vars.get("colormap", "inferno"))
    leg_fs = int(plot_vars.get("legend_fontsize", 8))

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    y_coords = np.linspace(0.0, ly, ny)
    for idx, sc in enumerate(scenarios):
        case_name = str(sc["name"])
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(results[case_name], origin="lower", extent=[0.0, lx, 0.0, ly], cmap=cmap_name)
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax_prof = axes[1, 1]
    for case_name, T_mat in results.items():
        ax_prof.plot(y_coords, T_mat[:, nx // 2], label=case_name)
    ax_prof.set_title("Centerline Profile T(x=Lx/2, y)")
    ax_prof.set_xlabel("y (m)")
    ax_prof.set_ylabel("Temperature (C)")
    ax_prof.legend(fontsize=leg_fs)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def _build_3d_kz_field(sc: dict, nz: int, ny: int, nx: int) -> np.ndarray:
    dist_type = str(sc["distribution"])
    k_uni = float(sc["k_uni"])
    k_bot = float(sc["k_bot"])
    k_top = float(sc["k_top"])

    if dist_type == "Uniform":
        return np.full((nz, ny, nx), k_uni, dtype=np.float64)

    z_norm = np.linspace(0.0, 1.0, nz)[:, None, None]
    if dist_type == "Linear":
        kz_1d = (1.0 - z_norm) * k_bot + z_norm * k_top
        return np.broadcast_to(kz_1d, (nz, ny, nx)).astype(np.float64).copy()

    num_layers = int(sc["num_layers"])
    inter_vals = [float(v) for v in sc["layer_k_Intermediate"]]
    layer_vals = [k_bot] + inter_vals + [k_top]
    bounds = np.linspace(0, nz, num_layers + 1, dtype=int)
    kz = np.empty((nz, ny, nx), dtype=np.float64)
    for m in range(num_layers):
        kz[bounds[m] : bounds[m + 1], :, :] = layer_vals[min(m, len(layer_vals) - 1)]
    return kz


def run_3d_simulation(spec_vars: dict, res_path: Path, csv_path: Path, plot_path: Path) -> None:
    grid_vars = spec_vars["grid_and_domain"]
    scenarios = spec_vars["comparison_scenarios"]
    plot_vars = spec_vars.get("plot_settings", {})

    nx = int(grid_vars["nx"])
    ny = int(grid_vars["ny"])
    nz = int(grid_vars["nz"])
    lx = float(grid_vars["lx"])
    ly = float(grid_vars["ly"])
    lz = float(grid_vars["lz"])
    kx = float(grid_vars["kx"])
    ky = float(grid_vars["ky"])
    t_bed = float(grid_vars["t_bed"])
    t_inf = float(grid_vars["t_inf"])
    h_conv = float(grid_vars["h_conv"])
    t_init_top = float(grid_vars["t_init_top_guess"])
    tol = float(grid_vars["tolerance"])
    max_iter = int(grid_vars["max_iter"])
    harmonic_eps = float(grid_vars["harmonic_eps"])

    dx = lx / float(nx - 1)
    dy = ly / float(ny - 1)
    dz = lz / float(nz - 1)

    saved_arrays = {}
    profiles = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["distribution", "top_mean_temp_c", "bottom_temp_c", "mid_z_temp_c", "iterations", "residual"])

        for sc in scenarios:
            dist_name = str(sc["name"])
            # Force full 3D shape (nz, ny, nx) for all conductivity distributions
            kz = np.broadcast_to(_build_3d_kz_field(sc, nz, ny, nx), (nz, ny, nx)).astype(np.float64).copy()

            T = np.linspace(t_bed, t_init_top, nz)[:, None, None] * np.ones((nz, ny, nx), dtype=np.float64)
            T[0, :, :] = t_bed

            # Pre-slice harmonic-mean conductivities on interior nodes (nz-2, ny-2, nx-2)
            k_center = kz[1:-1, 1:-1, 1:-1]
            k_up = kz[2:, 1:-1, 1:-1]
            k_down = kz[:-2, 1:-1, 1:-1]

            kz_plus = 2.0 * k_center * k_up / (k_center + k_up + harmonic_eps)
            kz_minus = 2.0 * k_center * k_down / (k_center + k_down + harmonic_eps)

            cx = kx / (dx * dx)
            cy = ky / (dy * dy)
            cz_p = kz_plus / (dz * dz)
            cz_m = kz_minus / (dz * dz)
            denom = 2.0 * cx + 2.0 * cy + cz_p + cz_m

            residual = 1.0
            it = 0
            for it in range(1, max_iter + 1):
                T_old = T.copy()
                T[1:-1, 1:-1, 1:-1] = (
                    cx * (T_old[1:-1, 1:-1, 2:] + T_old[1:-1, 1:-1, :-2])
                    + cy * (T_old[1:-1, 2:, 1:-1] + T_old[1:-1, :-2, 1:-1])
                    + cz_p * T_old[2:, 1:-1, 1:-1]
                    + cz_m * T_old[:-2, 1:-1, 1:-1]
                ) / denom

                # Adiabatic Neumann lateral walls (x=0, Lx; y=0, Ly)
                T[:, :, 0] = T[:, :, 1]
                T[:, :, -1] = T[:, :, -2]
                T[:, 0, :] = T[:, 1, :]
                T[:, -1, :] = T[:, -2, :]

                # Dirichlet bottom surface (z=0)
                T[0, :, :] = t_bed

                # Robin convective top surface (z=Lz)
                k_top_surf = kz[-1, :, :]
                T[-1, :, :] = (k_top_surf * T[-2, :, :] / dz + h_conv * t_inf) / (k_top_surf / dz + h_conv)

                residual = float(np.max(np.abs(T - T_old)))
                if residual < tol:
                    break

            saved_arrays[f"T_{dist_name.lower()}"] = T
            saved_arrays[f"kz_{dist_name.lower()}"] = kz
            profiles[dist_name] = T[:, ny // 2, nx // 2]

            top_mean = float(T[-1].mean())
            mid_z = float(T[nz // 2, ny // 2, nx // 2])
            writer.writerow([
                dist_name,
                f"{top_mean:.3f}",
                f"{float(T[0].mean()):.3f}",
                f"{mid_z:.3f}",
                it,
                f"{residual:.2e}",
            ])
            f.flush()
            np.savez_compressed(res_path, **saved_arrays)

    fig_w = float(plot_vars.get("figsize_width", 11))
    fig_h = float(plot_vars.get("figsize_height", 8))
    dpi = int(plot_vars.get("dpi", 140))
    cmap_name = str(plot_vars.get("colormap", "inferno"))
    leg_fs = int(plot_vars.get("legend_fontsize", 8))
    z_scale_mm = float(plot_vars.get("z_scale_mm", 1000.0))

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    z_coords_mm = np.linspace(0.0, lz * z_scale_mm, nz)
    for idx, sc in enumerate(scenarios):
        dist_name = str(sc["name"])
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(
            saved_arrays[f"T_{dist_name.lower()}"][:, ny // 2, :],
            origin="lower",
            aspect="auto",
            cmap=cmap_name,
        )
        ax.set_title(f"3D Wood XZ Cross-Section ({dist_name} kz)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax_p = axes[1, 1]
    for dist_name, prof in profiles.items():
        ax_p.plot(z_coords_mm, prof, marker="o", markersize=3, label=f"{dist_name} kz(z)")
    ax_p.set_title("Through-Thickness Profile T(z)")
    ax_p.set_xlabel("z (mm)")
    ax_p.set_ylabel("Temperature (C)")
    ax_p.legend(fontsize=leg_fs)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def main():
    spec_vars = load_simulation_variables("specs/simulation_spec.json")
    dims = int(spec_vars["spatial_dimensions"])

    res_path = Path(spec_vars["required_result_files"][0])
    csv_path = Path(spec_vars["required_data_files"][0])
    plot_path = Path(spec_vars["required_plots"][0])

    for p in (res_path, csv_path, plot_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    if dims == 1:
        run_1d_simulation(spec_vars, res_path, csv_path, plot_path)
    elif dims == 2:
        run_2d_simulation(spec_vars, res_path, csv_path, plot_path)
    elif dims == 3:
        run_3d_simulation(spec_vars, res_path, csv_path, plot_path)
    else:
        raise ValueError(f"Unsupported spatial_dimensions: {dims}")


if __name__ == "__main__":
    main()