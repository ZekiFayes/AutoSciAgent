# core/document_parser.py
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict


def parse_document(file_path: str | Path) -> str:
    """Extracts text from .md or .txt files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _parse_scalar_or_literal(raw_val: str) -> Any:
    val = raw_val.strip()
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def extract_variables_from_text(text: str) -> Dict[str, Any]:
    """
    Parses structured parameter blocks from `.md` or `.txt` content into named variables:
    1. Extracts fenced ```json ... ``` blocks.
    2. Extracts `variable_name = value` parameter assignments.
    """
    extracted: Dict[str, Any] = {}

    for match in re.finditer(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        try:
            parsed_json = json.loads(match.group(1))
            if isinstance(parsed_json, dict):
                extracted.update(parsed_json)
        except Exception:
            pass

    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or clean.startswith("//"):
            continue
        kv_match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$", clean)
        if kv_match:
            key, raw_val = kv_match.group(1), kv_match.group(2)
            extracted[key] = _parse_scalar_or_literal(raw_val)

    return extracted


def load_task_variables_from_folder(task_file_path: Path) -> Dict[str, Any]:
    """
    Reads `tasks/<task_id>/<task_id>.md` and any companion `tasks/<task_id>/*.txt` parameter files,
    returning all extracted physical and numerical parameters as a dictionary of variables.
    """
    variables: Dict[str, Any] = {}
    if task_file_path.exists():
        variables.update(extract_variables_from_text(parse_document(task_file_path)))

    task_dir = task_file_path.parent
    if task_dir.exists():
        for txt_file in sorted(task_dir.glob("*.txt")):
            if txt_file.is_file():
                variables.update(extract_variables_from_text(parse_document(txt_file)))

    return variables