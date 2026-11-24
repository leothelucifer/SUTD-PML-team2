"""With EnableSageMaker=false nothing in the deployment may need SageMaker access: no resource, no permission, no API call."""

import json
import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFN = ROOT / "infra" / "cloudformation"
sys.path.insert(0, str(ROOT / "scripts"))

import aws_published_model as published  # noqa: E402


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    tag = "Fn::" + suffix if suffix not in ("Ref", "Condition") else suffix
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    if suffix == "GetAtt" and isinstance(value, str):
        value = value.split(".", 1)
    return {tag: value}


_Loader.add_multi_constructor("!", _tag)
NOVALUE = object()


class Evaluator:
    """Just enough of CloudFormation's condition language to see which resources and statements a parameter set creates."""

    def __init__(self, template: dict, parameters: dict):
        self.template, self.values = template, {}
        for name, spec in template.get("Parameters", {}).items():
            self.values[name] = str(parameters.get(name, spec.get("Default", "")))
        self.expressions = template.get("Conditions", {})
        self._memo = {}
        self.conditions = self

    def __getitem__(self, name):
        if name not in self._memo:
            self._memo[name] = self.condition(self.expressions[name])
        return self._memo[name]

    def condition(self, node):
        (key, value), = node.items()
        if key == "Fn::Equals":
            return self.resolve(value[0]) == self.resolve(value[1])
        if key == "Fn::And":
            return all(self.condition(item) for item in value)
        if key == "Fn::Or":
            return any(self.condition(item) for item in value)
        if key == "Fn::Not":
            return not self.condition(value[0])
        if key == "Condition":
            return self[value]
        raise ValueError(key)

    def resolve(self, node):
        if isinstance(node, dict) and len(node) == 1:
            (key, value), = node.items()
            if key == "Ref":
                return NOVALUE if value == "AWS::NoValue" else self.values.get(value, f"<{value}>")
            if key == "Fn::If":
                chosen = value[1] if self.conditions[value[0]] else value[2]
                return self.resolve(chosen)
        if isinstance(node, dict):
            out = {k: self.resolve(v) for k, v in node.items()}
            return {k: v for k, v in out.items() if v is not NOVALUE}
        if isinstance(node, list):
            return [item for item in (self.resolve(i) for i in node) if item is not NOVALUE]
        return node

    def resources(self) -> dict:
        found = {}
        for name, body in self.template["Resources"].items():
            if "Condition" in body and not self.conditions[body["Condition"]]:
                continue
            found[name] = self.resolve({k: v for k, v in body.items() if k != "Condition"})
        return found


def load(name: str) -> dict:
    return yaml.load((CFN / name).read_text(encoding="utf-8"), Loader=_Loader)


DEV = json.loads((ROOT / "infra" / "params" / "dev.json").read_text(encoding="utf-8"))


class NoSageMakerTests(unittest.TestCase):
    def test_dev_turns_sagemaker_off_and_keeps_everything_on_ecs(self):
        self.assertEqual((DEV["EnableSageMaker"], DEV["TrainingBackend"], DEV["MlflowBackend"]), ("false", "ecs", "ecs"))

    def test_the_root_stack_creates_no_sagemaker_pipeline_and_no_sagemaker_mlops_parts(self):
        root = Evaluator(load("root.yaml"), DEV)
        created = root.resources()
        self.assertNotIn("Pipeline", created)                   # the SageMaker training pipeline
        for name in ("EcsPlatform", "PipelineEcs", "MlflowEcs", "Mlops"):
            self.assertIn(name, created)
        self.assertNotIn("SageMakerExecutionRoleArn", {k for k, v in load("root.yaml")["Outputs"].items()
                                                     if "Condition" not in v or root.conditions[v["Condition"]]})

    def test_the_mlops_template_creates_no_sagemaker_resource_or_permission(self):
        mlops = Evaluator(load("mlops.yaml"), {"EnableSageMaker": "false", "EnableMlflowApp": "true"})
        created = mlops.resources()
        text = json.dumps(created)
        self.assertNotIn("AWS::SageMaker", text)
        self.assertNotIn("sagemaker:", text)
        self.assertNotIn("sagemaker.amazonaws.com", text)
        self.assertNotIn("SageMakerExecutionRole", created)
        # The people's policies are still created, without the SageMaker statements.
        self.assertTrue({"TrainerPolicy", "ApproverPolicy", "DeployerPolicy"} <= set(created))

    def test_switching_it_on_brings_everything_back(self):
        on = Evaluator(load("mlops.yaml"), {"EnableSageMaker": "true", "EnableMlflowApp": "true"}).resources()
        self.assertIn("MlflowApp", on)
        self.assertIn("SageMakerExecutionRole", on)
        self.assertIn("sagemaker:CreateTrainingJob", json.dumps(on))
        only_training = Evaluator(load("mlops.yaml"), {"EnableSageMaker": "true", "EnableMlflowApp": "false"}).resources()
        self.assertNotIn("MlflowApp", only_training)
        self.assertIn("SageMakerExecutionRole", only_training)

    def test_the_ecs_templates_never_mention_sagemaker_apart_from_the_optional_app_permission(self):
        for name in ("ecs-platform.yaml", "pipeline-ecs.yaml", "mlflow-ecs.yaml", "evidently-ecs.yaml"):
            resolved = json.dumps(Evaluator(load(name), {}).resources())
            self.assertNotIn("sagemaker.amazonaws.com", resolved, name)
            self.assertNotIn("AWS::SageMaker", resolved, name)
        # With no SageMaker MLflow App passed in, the ECS roles carry no SageMaker permission either.
        platform = json.dumps(Evaluator(load("ecs-platform.yaml"), {"MlflowAppArn": ""}).resources())
        self.assertNotIn("sagemaker:", platform)

    def test_the_rule_stops_a_sagemaker_backend_without_the_switch(self):
        rules = load("root.yaml")["Rules"]
        self.assertIn("SageMakerBackendsNeedSageMaker", rules)
        text = json.dumps(rules["SageMakerBackendsNeedSageMaker"])
        self.assertIn("EnableSageMaker", text)

    def test_the_scripts_make_no_sagemaker_call_unless_the_stack_has_a_sagemaker_part(self):
        verify = (ROOT / "scripts" / "aws_verify.py").read_text(encoding="utf-8")
        self.assertRegex(verify, r'if outputs\.get\("MlflowAppArn"\):\n\s+check_mlflow')
        self.assertIn('outputs.get("TrainingBackend", "sagemaker") == "sagemaker"', verify)
        deploy = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        for command in ("cmd_train()", "cmd_mlflow_ui()", "cmd_promote()"):
            body = deploy.split(command, 1)[1].split("\n}\n", 1)[0]
            self.assertIn('[ -n "$(output MlflowAppArn)" ] ||', body, command)    # a clear message, not an API error
        publish = deploy.split("cmd_publish()", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("aws_published_model.py", publish)


class FakeS3:
    def __init__(self, objects):
        self.objects = objects     # key -> (metadata, body)

    def list_objects_v2(self, Bucket, Prefix, Delimiter, ContinuationToken=None):
        folders = sorted({re.match(r"(models/v\d+/)", k).group(1) for k in self.objects if re.match(r"models/v\d+/", k)})
        return {"CommonPrefixes": [{"Prefix": f} for f in folders], "IsTruncated": False}

    def head_object(self, Bucket, Key):
        return {"Metadata": self.objects[Key][0]}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception("NoSuchKey")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        return {"Body": type("B", (), {"read": lambda self_: self.objects[Key][1]})()}


class PublishedModelTests(unittest.TestCase):
    def s3(self, with_registry=True, sha="a" * 64):
        objects = {"models/v1/model.joblib": ({"sha256": "b" * 64}, b""), "models/v3/model.joblib": ({"sha256": sha}, b""),
                   "models/v10/model.joblib": ({"sha256": sha}, b"")}
        if with_registry:
            registry = json.dumps({"approved_by": "arn:aws:sts::1:assumed-role/approver/x", "frozen_evaluated_at": "2026-11-26T01:00:00+00:00"})
            objects["models/v10/registry.json"] = ({}, registry.encode())
        return FakeS3(objects)

    def test_the_newest_published_version_is_found_by_number_not_by_text(self):
        found = published.find(self.s3(), "bkt")
        self.assertEqual((found["version"], found["uri"], found["sha256"]), (10, "s3://bkt/models/v10/model.joblib", "a" * 64))
        self.assertEqual(published.find(self.s3(), "bkt", 3)["version"], 3)

    def test_missing_versions_or_hashes_are_refused_with_the_reason(self):
        with self.assertRaisesRegex(ValueError, "not published"):
            published.find(self.s3(), "bkt", 2)
        with self.assertRaisesRegex(ValueError, "no published model"):
            published.find(FakeS3({}), "bkt")
        with self.assertRaisesRegex(ValueError, "no SHA-256"):
            published.find(self.s3(sha=""), "bkt")

    def test_the_registry_record_is_read_for_the_approval_check(self):
        self.assertEqual(published.find(self.s3(), "bkt")["registry"]["approved_by"].split("/")[-2], "approver")
        self.assertEqual(published.find(self.s3(with_registry=False), "bkt")["registry"], {})


if __name__ == "__main__":
    unittest.main()
