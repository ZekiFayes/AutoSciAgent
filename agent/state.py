# agent/state.py
from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class SimulationSpec(BaseModel):
    task_title: str
    domain: str
    spatial_dimensions: Literal[1, 2, 3] = Field(default=2, description="1 for 1D PDEs, 2 for 2D PDEs, 3 for 3D PDEs.")
    governing_equations: str
    needs_derivation: bool = Field(default=True)
    grid_and_domain: Dict[str, Any] = Field(default_factory=dict)
    comparison_scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    plot_settings: Dict[str, Any] = Field(default_factory=dict)
    required_result_files: List[str] = Field(default_factory=lambda: ["outputs/results/simulation_fields.npz"])
    required_data_files: List[str] = Field(default_factory=lambda: ["outputs/data/summary_metrics.csv"])
    required_plots: List[str] = Field(default_factory=lambda: ["outputs/plots/comparison.png"])


class PreAuditOutput(BaseModel):
    is_feasible: bool = Field(description="True if 1D/2D/3D PDE is well-posed and fits within CPU/GPU RAM.")
    feasibility_notes: str
    feedback_for_requirement: str = ""


class PlanOutput(BaseModel):
    plan_steps: List[str]
    needs_derivation: bool
    selected_skills: List[str] = Field(default_factory=list)
    selected_library_entries: List[str] = Field(default_factory=list)
    updated_rolling_summary: str
    directive_for_next_node: str


class DerivationOutput(BaseModel):
    derivation_markdown: str
    stability_and_convergence_notes: str
    discretized_stencil_summary: str


class DerivationAuditOutput(BaseModel):
    is_valid: bool = Field(description="True if discretized stencil, BCs, and CFL stability bounds are mathematically valid.")
    math_and_bc_audit: str
    feedback_for_deriver: str = ""


class CodeGenerationOutput(BaseModel):
    explanation: str
    script_relative_path: str = Field(default="scripts/solver.py")
    python_code: str
    required_pip_packages: List[str] = Field(default_factory=list)


class DiagnoserOutput(BaseModel):
    status: Literal["converged_and_verified", "code_error", "derivation_error", "strategic_error"]
    diagnosis: str
    feedback_for_next_node: str


class ReporterOutput(BaseModel):
    report_markdown: str = Field(
        description="Complete quantitative Markdown engineering report comparing all simulated scenarios."
    )


class JudgePostAuditOutput(BaseModel):
    is_approved: bool
    physics_and_logic_audit: str
    target_node_if_rejected: Literal["planner", "solver_reporter"] = "solver_reporter"
    solver_entry_phase_if_rejected: Literal["derive", "code", "report"] = "code"
    planner_entry_phase_if_rejected: Literal["requirement", "plan"] = "plan"
    feedback_if_rejected: str = ""


class EditAttributionRow(BaseModel):
    component: str
    change: str
    observed: str
    verdict: Literal["helped", "hurt", "no observed effect", "confounded"]


class ReflectionOutput(BaseModel):
    edits: List[EditAttributionRow] = Field(default_factory=list)
    store_entry: bool
    cue: str
    lesson: str
    evidence: str
    confidence: Literal["high", "medium", "low"] = "high"


class AutoSciState(TypedDict, total=False):
    task_id: str
    task_file_path: str
    raw_task_text: str
    extracted_variables: Dict[str, Any]
    hardware_profile: Dict[str, Any]
    simulation_spec: Dict[str, Any]

    planner_entry_phase: Literal["requirement", "plan"]
    solver_entry_phase: Literal["derive", "code", "report"]

    plan_steps: List[str]
    needs_derivation: bool
    active_skills: List[str]
    active_library_entries: List[str]
    rolling_summary: str
    planner_directive: str

    derivation_path: str
    derivation_summary: str
    derivation_audit_notes: str

    generated_code_path: str
    required_packages: List[str]
    execution_status: str
    log_file_path: str
    log_tail: str
    stderr_traceback: str
    verification_report: Dict[str, Any]
    intermediate_results_digest: Dict[str, Any]
    generated_results: List[str]
    generated_plots: List[str]
    generated_data: List[str]
    report_path: str
    final_report_markdown: str

    req_retry_count: int
    inner_retry_count: int
    outer_loop_count: int
    inner_loop_trace: List[Dict[str, Any]]
    diagnostic_feedback: str
    judge_feedback: str
    library_entry_path: str
    next_node: str