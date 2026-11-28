import ast
import unittest
from pathlib import Path

RANK = {
    "data": 0,
    "features": 1,
    "models": 2,
    "training": 3,
    "evaluation": 3,
    "lifecycle": 4,
    "serving": 5,
    "monitoring": 5,
    "reporting": 6,
    "adapters": 7,
}


class ArchitectureTests(unittest.TestCase):
    def test_imports_point_left_in_the_dependency_direction(self):
        root = Path(__file__).resolve().parents[1] / "src" / "sepsis_warning"
        violations = []
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root)
            source_folder = relative.parts[0] if len(relative.parts) > 1 else "root"
            source_rank = RANK.get(source_folder)
            if source_rank is None:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            function_ranges = [
                (node.lineno, node.end_lineno)
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            for node in ast.walk(tree):
                if not hasattr(node, "lineno"):
                    continue
                if any(start <= node.lineno <= end for start, end in function_ranges):
                    continue
                module = node.module if isinstance(node, ast.ImportFrom) else None
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                else:
                    modules = []
                for imported in modules:
                    if imported.startswith("sepsis_warning."):
                        parts = imported.split(".")
                        if len(parts) < 3:
                            continue
                        target_folder = parts[1]
                    elif isinstance(node, ast.ImportFrom) and node.level:
                        resolved = []
                        base = list(relative.parts[:-1])
                        for _ in range(node.level - 1):
                            if base:
                                base.pop()
                        resolved.extend(base)
                        if node.module:
                            resolved.extend(node.module.split("."))
                        target_folder = resolved[0] if resolved and resolved[0] in RANK else source_folder
                    else:
                        continue
                    target_rank = RANK.get(target_folder)
                    if target_rank is not None and target_rank > source_rank:
                        violations.append(f"{relative}: imports {imported}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
