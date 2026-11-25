"""Every environment's parameter file runs the same ECS architecture and names only parameters the root template has."""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ECS_SETTINGS = {"TrainingBackend": "ecs", "MlflowBackend": "ecs", "EnableSageMaker": "false", "EnableSageMakerMlflowApp": "false",
                "EnableEvidently": "true", "EcsUseSpot": "true"}


def root_parameters() -> set[str]:
    text = (ROOT / "infra" / "cloudformation" / "root.yaml").read_text(encoding="utf-8")
    block = text[text.index("\nParameters:\n"):]
    block = block[:re.search(r"\n[A-Z][A-Za-z]*:\n", block[2:]).start() + 2]
    return set(re.findall(r"^  ([A-Za-z0-9]+):\s*$", block, flags=re.M))


class ParamsTests(unittest.TestCase):
    def files(self):
        return {name: json.loads((ROOT / "infra" / "params" / f"{name}.json").read_text(encoding="utf-8")) for name in ("dev", "staging", "prod")}

    def test_every_key_is_a_parameter_of_the_root_template(self):
        known = root_parameters()
        self.assertIn("EcsUseSpot", known)
        for name, params in self.files().items():
            unknown = {k for k in params if not k.startswith("_")} - known
            self.assertEqual(unknown, set(), name)

    def test_staging_and_prod_run_on_ecs_like_dev(self):
        """They once set none of this and would have deployed the SageMaker architecture, which needs a quota."""
        for name, params in self.files().items():
            for key, value in ECS_SETTINGS.items():
                self.assertEqual(params.get(key), value, f"{name}: {key}")

    def test_the_environment_name_matches_the_file(self):
        for name, params in self.files().items():
            self.assertEqual(params["EnvironmentName"], f"sepsis-{name}")


if __name__ == "__main__":
    unittest.main()
