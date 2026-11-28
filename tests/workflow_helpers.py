"""Helpers for asserting workflow behavior through local composite actions."""

import copy
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
INPUT_EXPRESSION = re.compile(r"\$\{\{\s*inputs\.([A-Za-z0-9_-]+)\s*\}\}")


def load_workflow(name: str) -> dict:
    document = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    document["on"] = document.pop(True, document.get("on"))
    return document


def _expand_inputs(value, inputs):
    if isinstance(value, str):
        return INPUT_EXPRESSION.sub(lambda match: str(inputs.get(match.group(1), match.group(0))), value)
    if isinstance(value, list):
        return [_expand_inputs(item, inputs) for item in value]
    if isinstance(value, dict):
        return {key: _expand_inputs(item, inputs) for key, item in value.items()}
    return value


def workflow_steps(name: str, job_name: str | None = None) -> list[dict]:
    """Return workflow steps with local composite-action steps expanded in place."""
    workflow = load_workflow(name)
    jobs = workflow["jobs"]
    selected = [jobs[job_name]] if job_name else jobs.values()
    result = []
    for job in selected:
        for step in job.get("steps", []):
            local_action = step.get("uses", "")
            if not local_action.startswith("./.github/actions/"):
                result.append(step)
                continue
            action_path = ROOT / local_action.removeprefix("./")
            action = yaml.safe_load((action_path / "action.yml").read_text(encoding="utf-8"))
            inputs = {
                key: spec.get("default", "")
                for key, spec in action.get("inputs", {}).items()
            }
            inputs.update(step.get("with", {}))
            for action_step in action["runs"]["steps"]:
                expanded = _expand_inputs(copy.deepcopy(action_step), inputs)
                expanded["_composite_action"] = local_action
                expanded["_outer_step_id"] = step.get("id")
                if step.get("name"):
                    expanded["name"] = f"{step['name']} / {expanded.get('name', '')}"
                result.append(expanded)
    return result
