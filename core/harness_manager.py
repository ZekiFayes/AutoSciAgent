# core/harness_manager.py
import py_compile
import re
from pathlib import Path
from typing import Dict, List, Tuple
from core import PROJECT_ROOT, load_config
from core.logger import setup_logger

logger = setup_logger("Harness-Manager", mode="system")


class HarnessManager:
    """
    Manages the externalized HARNESS² directory (`harness/`), phase-sliced node prompts
    (`harness/nodes/{planner,solver_reporter,judge_reflector}.md`), and the priority lookup:
      1st: `harness/skills/*/SKILL.md`
      2nd: `library/INDEX.md` -> `library/<task_id>/EXPERIENCE.md` & `skills_snapshot/`
    """

    def __init__(self, harness_dir: Path | None = None):
        cfg = load_config()
        rel_h = cfg["paths"]["harness_dir"]
        self.harness_dir = (harness_dir or (PROJECT_ROOT / rel_h)).resolve()
        self.library_dir = (PROJECT_ROOT / cfg["paths"]["library_dir"]).resolve()
        self.max_always_on_chars = int(cfg["agent"]["rrsi_max_always_on_chars"])

    def load_global_always_on_harness(self) -> str:
        files = [
            ("C1: SYSTEM PROMPT", self.harness_dir / "systemprompt.md"),
            ("C2: GUARDRAILS", self.harness_dir / "guiderails.md"),
            ("C3: LONG-TERM MEMORY", self.harness_dir / "LongTermMEMORY.md"),
            ("C4: TOOL GUIDANCE", self.harness_dir / "toolguidance.md"),
        ]
        blocks = []
        for label, path in files:
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    blocks.append(f"### [{label}]\n{text}")
        return "\n\n".join(blocks)

    def load_node_prompt(self, node_name: str, runtime_context: dict) -> str:
        """
        Mounts the node `.md` file raw (safe for LaTeX braces) and slices out only the
        active `<!-- PHASE:<node_phase> --> ... <!-- END_PHASE:<node_phase> -->` section when provided.
        """
        global_ctx = self.load_global_always_on_harness()
        node_file = self.harness_dir / "nodes" / f"{node_name}.md"
        node_instructions = (
            node_file.read_text(encoding="utf-8").strip()
            if node_file.exists()
            else f"# ROLE: {node_name.upper()}"
        )

        active_phase = runtime_context.get("node_phase")
        if active_phase:
            pattern = rf"<!-- PHASE:{re.escape(str(active_phase))} -->(.*?)<!-- END_PHASE:{re.escape(str(active_phase))} -->"
            match = re.search(pattern, node_instructions, flags=re.DOTALL)
            if match:
                node_instructions = f"# ROLE: {node_name.upper()} (PHASE: {active_phase})\n\n{match.group(1).strip()}"

        context_lines = [
            f"### [{k.upper()}]\n{v}"
            for k, v in runtime_context.items()
            if v is not None and v != ""
        ]
        runtime_block = "\n\n".join(context_lines)
        return f"{global_ctx}\n\n{node_instructions}\n\n---\n## RUNTIME STATE & CONTEXT\n\n{runtime_block}"

    def list_available_skills(self) -> Dict[str, str]:
        """Priority 1 Lookup: Inspects `harness/skills/*/SKILL.md` (returns `{}` if empty)."""
        skills_dir = self.harness_dir / "skills"
        if not skills_dir.exists():
            return {}
        found: Dict[str, str] = {}
        for sdir in sorted(skills_dir.iterdir()):
            skill_md = sdir / "SKILL.md"
            if sdir.is_dir() and skill_md.exists():
                content = skill_md.read_text(encoding="utf-8").strip()
                if content:
                    found[sdir.name] = content
        return found

    def load_skill(self, skill_name: str) -> str:
        p = self.harness_dir / "skills" / skill_name / "SKILL.md"
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""

    def load_library_index(self) -> str:
        """Priority 2 Lookup: Inspects `library/INDEX.md` (returns `""` if empty)."""
        idx = self.library_dir / "INDEX.md"
        if not idx.exists():
            return ""
        lines = [
            ln.strip()
            for ln in idx.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        return "\n".join(lines) if lines else ""

    def list_library_entries(self) -> List[str]:
        if not self.library_dir.exists():
            return []
        return [
            d.name
            for d in sorted(self.library_dir.iterdir())
            if d.is_dir() and (d / "EXPERIENCE.md").exists()
        ]

    def load_library_experience(self, entry_id: str) -> str:
        """
        Loads `library/<entry_id>/EXPERIENCE.md` and any companion `skills_snapshot/*/SKILL.md`
        so lessons learned in earlier tasks can be applied in Step 2.1 and Step 2.3.
        """
        entry_dir = self.library_dir / entry_id
        exp_file = entry_dir / "EXPERIENCE.md"
        if not exp_file.exists():
            return ""

        parts = [f"### [LIBRARY EXPERIENCE: {entry_id}]\n{exp_file.read_text(encoding='utf-8').strip()}"]
        snap_dir = entry_dir / "skills_snapshot"
        if snap_dir.exists():
            for sdir in sorted(snap_dir.iterdir()):
                smd = sdir / "SKILL.md"
                if sdir.is_dir() and smd.exists():
                    parts.append(f"#### [SNAPSHOT SKILL: {sdir.name}]\n{smd.read_text(encoding='utf-8').strip()}")
        return "\n\n".join(parts)

    def validate_harness_tree(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        sys_p = self.harness_dir / "systemprompt.md"
        guard_p = self.harness_dir / "guiderails.md"
        sys_txt = sys_p.read_text(encoding="utf-8") if sys_p.exists() else ""
        guard_txt = guard_p.read_text(encoding="utf-8") if guard_p.exists() else ""
        combined_triggers = sys_txt + "\n" + guard_txt

        skills_dir = self.harness_dir / "skills"
        if skills_dir.exists():
            for sdir in sorted(skills_dir.iterdir()):
                if sdir.is_dir() and not sdir.name.startswith("."):
                    if not (sdir / "SKILL.md").exists():
                        errors.append(f"Rule 1 Violation: Skill '{sdir.name}' is missing SKILL.md.")
                    elif sdir.name not in combined_triggers:
                        errors.append(
                            f"Rule 1 Violation: Skill '{sdir.name}' has no trigger line in systemprompt.md or guiderails.md."
                        )

        total_chars = 0
        for fname in ("systemprompt.md", "guiderails.md", "LongTermMEMORY.md", "toolguidance.md"):
            fpath = self.harness_dir / fname
            if fpath.exists():
                txt = fpath.read_text(encoding="utf-8")
                total_chars += len(txt)
                if "{{" in txt or "{%" in txt:
                    errors.append(f"Rule 2 Violation: Template braces '{{{{' or '{{%' found in {fname}.")

        if total_chars > self.max_always_on_chars:
            errors.append(
                f"RRSI Regularization Violation: Always-on C1-C4 files total {total_chars} chars "
                f"(cap is {self.max_always_on_chars})."
            )

        for script_rel in set(re.findall(r"scripts/([a-zA-Z0-9_\-]+\.py)", combined_triggers)):
            sp = self.harness_dir / "scripts" / script_rel
            if not sp.exists():
                errors.append(f"Rule 3 Violation: Referenced script 'scripts/{script_rel}' does not exist.")
            else:
                try:
                    py_compile.compile(str(sp), doraise=True)
                except py_compile.PyCompileError as exc:
                    errors.append(f"Rule 3 Violation: Script 'scripts/{script_rel}' failed syntax check: {exc}")

        return len(errors) == 0, errors