# core/__init__.py
import json
import os
import re
from pathlib import Path
from typing import Any, Type
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_solver_template_code() -> str:
    """Loads `harness/scripts/solver_template.py` from disk so no solver code is hardcoded in `core/`."""
    cfg = load_config()
    harness_rel = cfg["paths"]["harness_dir"]
    template_path = PROJECT_ROOT / harness_rel / "scripts" / "solver_template.py"
    return template_path.read_text(encoding="utf-8")


class _DeterministicStructuredMock:
    """
    Offline deterministic responder used when AUTOSCI_TEST_MODE=1.
    Reads task variables, skills, and library entries dynamically from the prompt context.
    """

    def __init__(self, schema: Type[Any]):
        self.schema = schema

    @staticmethod
    def _extract_vars_from_prompt(prompt: str) -> dict:
        match = re.search(r"### \[EXTRACTED_TASK_VARIABLES\]\s*(\{.*?\})\s*(?:###|\Z)", prompt, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        for block in re.finditer(r"```json\s*(\{.*?\})\s*```", prompt, flags=re.DOTALL):
            try:
                return json.loads(block.group(1))
            except Exception:
                pass
        return {}

    def invoke(self, prompt: str) -> Any:
        cfg = load_config()
        paths_cfg = cfg["paths"]
        name = self.schema.__name__
        task_vars = self._extract_vars_from_prompt(prompt)

        if name == "SimulationSpec":
            return self.schema(
                task_title=task_vars.get("task_title", "PDE Simulation Task"),
                domain=task_vars.get("domain", "general_pde"),
                spatial_dimensions=int(task_vars.get("spatial_dimensions", 2)),
                governing_equations=task_vars.get("governing_equations", "div(K grad T) = 0"),
                needs_derivation=bool(task_vars.get("needs_derivation", True)),
                grid_and_domain=task_vars.get("grid_and_domain", {}),
                comparison_scenarios=task_vars.get("comparison_scenarios", []),
                plot_settings=task_vars.get("plot_settings", {}),
                required_result_files=task_vars.get("required_result_files", [paths_cfg["results_relative_path"]]),
                required_data_files=task_vars.get("required_data_files", [paths_cfg["metrics_relative_path"]]),
                required_plots=task_vars.get("required_plots", [paths_cfg["plot_relative_path"]]),
            )
        if name == "PreAuditOutput":
            return self.schema(
                is_feasible=True,
                feasibility_notes="1D/2D/3D PDE and boundary conditions are well-posed; grid memory fits within available RAM.",
                feedback_for_requirement="Approved.",
            )
        if name == "PlanOutput":
            found_skills = re.findall(r"### Skill `([a-zA-Z0-9_\-]+)`", prompt)
            found_lib = re.findall(r"\*\*\[`([a-zA-Z0-9_\-]+)`\]", prompt)
            return self.schema(
                plan_steps=[
                    "Check active SKILL.md first, then library/INDEX.md second (skipping if empty or mismatched).",
                    "Derive and audit discretized PDE stencil and boundary condition equations.",
                    "Generate variable-driven solver reading specs/simulation_spec.json.",
                ],
                needs_derivation=True,
                selected_skills=found_skills,
                selected_library_entries=found_lib,
                updated_rolling_summary="Checked SKILL.md and library/INDEX.md; routing to Unified Solver & Reporter.",
                directive_for_next_node="Derive, audit, code, execute, and synthesize final report.",
            )
        if name == "DerivationOutput":
            return self.schema(
                derivation_markdown=(
                    "# Conservative 1D/2D/3D PDE Discretization\n\n"
                    "Derived 1D FTCS (3-point), 2D orthotropic (5-point), and 3D harmonic-mean variable-conductivity "
                    "(7-point) stencils with Dirichlet, Neumann, and Robin boundaries parameterized by spec variables."
                ),
                stability_and_convergence_notes="Satisfies Courant-Friedrichs-Lewy Fourier stability and Discrete Maximum Principle.",
                discretized_stencil_summary="Variable-driven 1D/2D/3D conservative stencil reading specs/simulation_spec.json.",
            )
        if name == "DerivationAuditOutput":
            return self.schema(
                is_valid=True,
                math_and_bc_audit="Verified stencil conservation weights, boundary signs, and CFL stability criteria.",
                feedback_for_deriver="",
            )
        if name == "CodeGenerationOutput":
            return self.schema(
                explanation="Variable-driven 1D/2D/3D solver reading all parameters from specs/simulation_spec.json.",
                script_relative_path=paths_cfg["solver_relative_path"],
                python_code=load_solver_template_code(),
                required_pip_packages=[],
            )
        if name == "DiagnoserOutput":
            return self.schema(
                status="converged_and_verified",
                diagnosis="Solver converged; .npz fields, .csv metrics, and .png plots passed verification.",
                feedback_for_next_node="Proceed to Report Synthesis Phase.",
            )
        if name == "ReporterOutput":
            return self.schema(
                report_markdown=(
                    "# Final Engineering Report\n\n"
                    "## 1. Variable-Driven Mathematical & Numerical Formulation\n"
                    "Loaded all domain, mesh, boundary, and conductivity variables from `specs/simulation_spec.json` "
                    "and saved full field arrays in `outputs/results/simulation_fields.npz`.\n\n"
                    "## 2. Quantitative Scenario Comparison\n"
                    "All scenarios converged with zero `NaN`/`Inf` values (see `outputs/data/summary_metrics.csv` "
                    "and `outputs/plots/comparison.png`).\n"
                )
            )
        if name == "JudgePostAuditOutput":
            return self.schema(
                is_approved=True,
                physics_and_logic_audit="Physical boundary laws, conservation, and .npz/.csv/.png/.md deliverables are consistent.",
                target_node_if_rejected="solver_reporter",
                solver_entry_phase_if_rejected="code",
                planner_entry_phase_if_rejected="plan",
                feedback_if_rejected="",
            )
        if name == "ReflectionOutput":
            return self.schema(
                edits=[
                    {
                        "component": "skills/numerical_pde/SKILL.md",
                        "change": "Applied variable-driven 1D/2D/3D discretization and verified .npz/.csv/.png outputs",
                        "observed": "Converged cleanly across all parameter sweep scenarios",
                        "verdict": "helped",
                    }
                ],
                store_entry=True,
                cue="1D/2D/3D heat conduction with variable-driven parameters from .md files",
                lesson="Read all physical and mesh parameters into variables from specs/simulation_spec.json and inspect .npz slices via scripts/inspect_results.py.",
                evidence="Verified across 1D, 2D, and 3D tasks with zero NaNs.",
                confidence="high",
            )
        raise ValueError(f"Unsupported mock schema: {name}")


class _DeterministicChatLLM:
    def with_structured_output(self, schema: Type[Any]):
        return _DeterministicStructuredMock(schema)

    def invoke(self, prompt: str):
        class _Msg:
            content = (
                "# Final Engineering Report\n\n"
                "Completed simulation and verified `outputs/results/simulation_fields.npz`, "
                "`outputs/data/summary_metrics.csv`, and `outputs/plots/comparison.png`.\n"
            )
        return _Msg()


def get_llm(temperature: float | None = None, model_id: str | None = None) -> Any:
    if os.getenv("AUTOSCI_TEST_MODE", "0") == "1":
        return _DeterministicChatLLM()

    from langchain_google_genai import ChatGoogleGenerativeAI
    cfg = load_config()
    llm_cfg = cfg["llm"]
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_gemini_api_key_here":
        raise EnvironmentError("Valid GEMINI_API_KEY is missing in .env (or set AUTOSCI_TEST_MODE=1 for offline tests).")
    return ChatGoogleGenerativeAI(
        model=model_id or llm_cfg["model_id"],
        temperature=temperature if temperature is not None else float(llm_cfg["temperature"]),
        google_api_key=api_key,
    )