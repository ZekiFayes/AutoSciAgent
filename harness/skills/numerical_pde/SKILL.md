---
name: numerical_pde
description: 1D (3-point), 2D (5-point), and 3D (7-point) steady/transient PDE stencils with variable-driven parameters and Dirichlet, Neumann, and Robin boundaries.
---
# Skill: Variable-Driven 1D, 2D & 3D Finite Difference / Finite Volume PDE Solver

## 1. Load Parameters from `specs/simulation_spec.json` into Variables
Always load `spec = json.loads(Path("specs/simulation_spec.json").read_text(encoding="utf-8"))` and unpack `grid_and_domain`, `comparison_scenarios`, and `plot_settings` into named variables instead of hardcoding numbers.

## 2. 1D Transient Heat Conduction Stencil (3-Point FTCS)
For $\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$ with Fourier number $\text{Fo} = \frac{\alpha \Delta t}{(\Delta x)^2} \le \text{Fo}_{\max}$:
`T_new[1:-1] = T[1:-1] + fo * (T[2:] - 2.0 * T[1:-1] + T[:-2])`

## 3. 2D Orthotropic Stencil (5-Point, $\Delta x = \Delta y$)
For $k_x \partial_{xx} T + k_y \partial_{yy} T = 0$:
`T_new[1:-1, 1:-1] = (kx * (T[1:-1, 2:] + T[1:-1, :-2]) + ky * (T[2:, 1:-1] + T[:-2, 1:-1])) / (2.0 * (kx + ky))`

## 4. 3D Spatially Varying Conductivity Stencil (7-Point Harmonic-Mean)
For $\nabla \cdot (\mathbf{K}(x,y,z)\nabla T) = 0$:
`kz_plus = 2.0 * kz[1:-1] * kz[2:] / (kz[1:-1] + kz[2:] + harmonic_eps)`
- **Dirichlet at $z=0$**: `T[0, :, :] = t_bed`
- **Robin Convection at $z=L_z$**: `T[-1, :, :] = (k_top_surf * T[-2, :, :] / dz + h_conv * t_inf) / (k_top_surf / dz + h_conv)`
- **Adiabatic Neumann at $x=0, L_x$ and $y=0, L_y$**: `T[:, :, 0] = T[:, :, 1]; T[:, :, -1] = T[:, :, -2]; T[:, 0, :] = T[:, 1, :]; T[:, -1, :] = T[:, -2, :]`