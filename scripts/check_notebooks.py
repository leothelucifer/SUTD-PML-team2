#!/usr/bin/env python3
"""Keep notebooks as thin views over the package (run in CI).

Fails when a notebook:
* has stored outputs (they can hold patient rows and make diffs unreadable), or
* re-implements feature or preprocessing logic instead of importing
  `sepsis_warning` (a second copy drifts from the code that training and
  serving share).

    python scripts/check_notebooks.py [paths...]     # default: every tracked *.ipynb
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED_FOLDERS = {".git", ".venv", "venv", "env", "node_modules", "artifacts", "data", ".ipynb_checkpoints"}

# Carry-forward, back-fill and windowing are what the model's features are made of; a second
# copy belongs in sepsis_warning/features.py. Exploratory scikit-learn use (PCA, a quick
# cross-validated profile) is not flagged.
PREPROCESSING = re.compile(r"\.(ffill|bfill|fillna|interpolate|rolling|ewm)\(")
DEFINES_FEATURES = re.compile(r"^\s*def\s+\w*(feature|impute|preprocess|window)\w*\(", re.MULTILINE | re.IGNORECASE)


def problems(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = []
    code = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    with_outputs = sum(bool(cell.get("outputs")) for cell in code)
    if with_outputs:
        found.append(f"{with_outputs} cell(s) have stored outputs; clear them before committing")
    for number, cell in enumerate(code, start=1):
        source = "".join(cell.get("source", []))
        if DEFINES_FEATURES.search(source):
            found.append(f"code cell {number} defines a feature or preprocessing function; move it into the package")
        match = PREPROCESSING.search(source)
        if match:
            found.append(f"code cell {number} uses {match.group(0).strip('.(')}; "
                         "use sepsis_warning.features so training and serving stay identical")
    return found


def default_notebooks() -> list[Path]:
    """Every tracked notebook; when Git lists none (a downloaded zip has no checkout), every notebook under the project."""
    try:
        listed = subprocess.run(["git", "ls-files", "*.ipynb"], capture_output=True, text=True, check=False, cwd=ROOT)
        tracked = [ROOT / line for line in listed.stdout.splitlines() if line] if listed.returncode == 0 else []
    except FileNotFoundError:  # Git is not installed
        tracked = []
    return tracked or sorted(path for path in ROOT.rglob("*.ipynb")
                             if not SKIPPED_FOLDERS & set(path.relative_to(ROOT).parts[:-1]))


def main(argv=None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        paths = default_notebooks()
    failed = False
    for path in paths:
        for problem in problems(path):
            failed = True
            print(f"{path}: {problem}")
    if not failed:
        print(f"{len(paths)} notebook(s) checked: no stored outputs, no duplicated preprocessing")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
