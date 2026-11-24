"""The predict demo: freeze rule, replay arguments and the workflow. No AWS."""

import datetime
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aws_demo as demo  # noqa: E402


class DemoTests(unittest.TestCase):
    def test_held_out_splits_need_a_recorded_frozen_evaluation(self):
        for split in ("internal_test", "external_test"):
            for registry in (None, {}, {"frozen_evaluated_at": None}):
                with self.assertRaisesRegex(ValueError, "no recorded frozen evaluation"):
                    demo.require_frozen(split, registry)
            demo.require_frozen(split, {"frozen_evaluated_at": "2026-11-25T01:00:00+00:00"})
        demo.require_frozen("validation", None)          # validation was used for selection: always allowed
        demo.require_frozen("train", {})

    def test_the_replay_command_scores_through_the_signed_endpoint_and_marks_the_freeze_for_held_out_splits(self):
        base = dict(api_url="https://x.lambda-url.r.on.aws", data=Path("d.parquet"), splits=Path("s.csv"), beds=12,
                    seed=42, output=Path("w.html"))
        validation = demo.ward_argv(split="validation", **base)
        self.assertIn("--sign", validation)
        self.assertEqual(validation[validation.index("--api-url") + 1], "https://x.lambda-url.r.on.aws")
        self.assertNotIn("--after-freeze", validation)
        self.assertIn("--after-freeze", demo.ward_argv(split="external_test", **base))

    def test_the_page_key_is_under_the_private_demo_prefix(self):
        key = demo.report_key(datetime.datetime(2026, 11, 25, 3, 4, 5, tzinfo=datetime.UTC), "validation")
        self.assertEqual(key, "reports/demo/20261125T030405Z/ward-validation.html")

    def test_a_missing_registry_file_is_an_answer_not_a_crash(self):
        class S3:
            def get_object(self, Bucket, Key):
                error = Exception("NoSuchKey")
                error.response = {"Error": {"Code": "NoSuchKey"}}
                raise error

        self.assertIsNone(demo.read_json(S3(), "b", "models/v1/registry.json"))

    def test_the_workflow_is_manual_keeps_inputs_out_of_scripts_and_never_uploads_the_page_as_an_artifact(self):
        text = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})
        self.assertEqual(workflow[True]["workflow_dispatch"]["inputs"]["split"]["options"], ["validation", "internal_test", "external_test"])
        self.assertNotIn("upload-artifact", text)
        for step in workflow["jobs"]["demo"]["steps"]:
            self.assertNotIn("${{ inputs.", step.get("run", ""))
        self.assertIn("demo) cmd_demo", (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
