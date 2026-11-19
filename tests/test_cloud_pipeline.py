"""The cloud pipeline definition, its start script and workflow: shape checks that need no AWS."""

import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):  # CloudFormation tags (!Ref, !Sub, ...) become plain values
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_Loader.add_multi_constructor("!", _tag)


def definition() -> dict:
    template = yaml.load((ROOT / "infra" / "cloudformation" / "pipeline.yaml").read_text(encoding="utf-8"), Loader=_Loader)
    return template["Resources"]["TrainingPipeline"]["Properties"]["Definition"]


class PipelineTests(unittest.TestCase):
    def test_the_states_run_in_order(self):
        machine = definition()
        self.assertEqual(machine["StartAt"], "CheckPrepare")
        states = machine["States"]
        self.assertEqual(states["CheckPrepare"]["Default"], "Tune")
        self.assertEqual(states["CheckPrepare"]["Choices"][0]["Next"], "PrepareData")
        chain = ["PrepareData", "Tune", "Train", "WaitForApproval", "EvaluateAndPublish", "NotifyPublished"]
        for current, following in zip(chain, chain[1:], strict=False):
            self.assertEqual(states[current]["Next"], following, current)
        for name, state in states.items():
            if state["Type"] == "Task" and name not in ("NotifyFailure", "NotifyPublished"):
                self.assertEqual(state["Catch"][0]["Next"], "NotifyFailure", name)

    def test_modes_and_channels(self):
        states = definition()["States"]
        self.assertEqual(states["PrepareData"]["Parameters"]["HyperParameters"]["mode"], "prepare")
        self.assertNotIn("InputDataConfig", states["PrepareData"]["Parameters"])  # it makes the data
        self.assertEqual(states["Tune"]["Parameters"]["HyperParameters"]["mode"], "tune")
        channels = {c["ChannelName"]: c for c in states["Train"]["Parameters"]["InputDataConfig"]}
        self.assertEqual(set(channels), {"data", "config"})
        self.assertIn("tuning/", channels["config"]["DataSource"]["S3DataSource"]["S3Uri.$"])
        self.assertNotIn("EnableManagedSpotTraining", states["PrepareData"]["Parameters"])  # a download must not restart

    def test_states_format_has_one_argument_per_placeholder(self):
        # A mismatch is a runtime error that only shows when the pipeline reaches that step.
        for name, state in definition()["States"].items():
            for key, value in state.get("Parameters", {}).items():
                if isinstance(value, str) and "States.Format(" in value:
                    inner = value[value.index("States.Format(") + len("States.Format("):].rstrip().rstrip(")")
                    template, arguments = re.match(r"'(.*)'\s*,\s*(.*)$", inner, flags=re.S).groups()
                    self.assertEqual(template.count("{}"), len([a for a in arguments.split(",") if a.strip()]), f"{name}.{key}")

    def test_the_training_role_can_write_the_pipeline_artifacts(self):
        text = (ROOT / "infra" / "cloudformation" / "mlops.yaml").read_text(encoding="utf-8")
        for prefix in ("data", "tuning", "reports"):
            self.assertIn(f'${{ArtifactBucketArn}}/{prefix}/*', text)
        self.assertIn("WritePipelineArtifacts", text)


class StartScriptTests(unittest.TestCase):
    def run_script(self, **env):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            fake = temp / "aws"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"$1\" = cloudformation ]; then echo arn:aws:states:r:1:stateMachine:m; exit 0; fi\n"
                "if [ \"$1\" = stepfunctions ] && [ \"$2\" = start-execution ]; then\n"
                "  while [ $# -gt 0 ]; do [ \"$1\" = --input ] && printf '%s' \"$2\" > \"$CAPTURE\"; shift; done\n"
                "  echo arn:aws:states:r:1:execution:m:e; exit 0; fi\n"
                "exit 0\n")
            fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
            capture = temp / "input.json"
            full_env = {**os.environ, "PATH": f"{temp}:{os.environ['PATH']}", "AWS_REGION": "ap-southeast-2",
                        "ENVIRONMENT": "dev", "TAG": "abc123", "CAPTURE": str(capture), **env}
            result = subprocess.run(["bash", str(ROOT / "scripts" / "aws_deploy.sh"), "pipeline"], env=full_env,
                                    capture_output=True, text=True, cwd=ROOT)
            return result, json.loads(capture.read_text()) if capture.exists() else None

    def test_defaults_are_a_synthetic_rehearsal(self):
        result, sent = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({k: sent[k] for k in ("image_tag", "prepare", "data_source", "limit", "trials", "require_real_data")},
                         {"image_tag": "training-abc123", "prepare": "true", "data_source": "synthetic", "limit": "0",
                          "trials": "6", "require_real_data": "false"})

    def test_real_data_requires_real_data_and_reuse_is_possible(self):
        result, sent = self.run_script(DATA_SOURCE="physionet", PREPARE="false", LIMIT="500", TRIALS="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((sent["require_real_data"], sent["prepare"], sent["limit"], sent["trials"]), ("true", "false", "500", "0"))

    def test_bad_values_are_refused_before_anything_starts(self):
        for env in ({"DATA_SOURCE": "other"}, {"PREPARE": "maybe"}, {"TRIALS": "6; rm -rf /"}, {"LIMIT": "-1"}):
            result, sent = self.run_script(**env)
            self.assertNotEqual(result.returncode, 0, env)
            self.assertIsNone(sent, env)


class WorkflowTests(unittest.TestCase):
    def test_pipeline_workflow(self):
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "pipeline.yml").read_text(encoding="utf-8"))
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["data_source"]["options"], ["synthetic", "physionet"])
        self.assertEqual(inputs["trials"]["default"], "6")
        self.assertEqual(workflow["permissions"]["id-token"], "write")
        names = [s.get("name", "") for s in workflow["jobs"]["pipeline"]["steps"]]
        self.assertLess(names.index("Build the images for this commit (skipped when they already exist)"),
                        names.index("Run the pipeline and follow it"))
        self.assertEqual(workflow["jobs"]["pipeline"]["environment"], "${{ inputs.environment }}")


if __name__ == "__main__":
    unittest.main()
