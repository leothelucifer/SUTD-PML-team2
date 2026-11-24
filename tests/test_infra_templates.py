"""CloudFormation rules that cfn-lint does not catch but the service rejects at create time."""

import re
import unittest
from pathlib import Path

TEMPLATES = sorted((Path(__file__).resolve().parents[1] / "infra").rglob("*.yaml"))


def rules_block(text: str) -> str:
    match = re.search(r"^Rules:\n((?:[ \t].*\n|\n)*)", text, flags=re.M)
    return match.group(1) if match else ""


class TemplateTests(unittest.TestCase):
    def test_rules_blocks_do_not_use_the_condition_function(self):
        self.assertTrue(TEMPLATES)
        for path in TEMPLATES:
            block = rules_block(path.read_text(encoding="utf-8"))
            self.assertNotIn("!Condition", block, f"{path.name}: Fn::Condition is not supported in Rules")
            self.assertNotIn("Fn::Condition", block, f"{path.name}: Fn::Condition is not supported in Rules")

    def test_no_yaml_null_values(self):
        # CloudFormation rejects "'null' values are not allowed in templates" (for example ResultPath: null
        # in a Step Functions definition) when the stack is created, and cfn-lint does not.
        pattern = re.compile(r"^[^#\n]*:\s*(null|~|Null|NULL)\s*(#.*)?$", re.M)
        for path in TEMPLATES:
            found = pattern.findall(path.read_text(encoding="utf-8"))
            self.assertEqual(found, [], f"{path.name} has a YAML null value")

    def test_logged_bucket_waits_for_the_log_bucket_policy(self):
        text = (Path(__file__).resolve().parents[1] / "infra" / "cloudformation" / "artifacts.yaml").read_text(encoding="utf-8")
        bucket = text.split("  ArtifactBucket:\n", 1)[1].split("\n  ArtifactBucketPolicy:", 1)[0]
        self.assertIn("LoggingConfiguration", bucket)
        self.assertIn("DependsOn: AccessLogBucketPolicy", bucket)

    def test_retained_artifacts_use_cloudformation_managed_names(self):
        text = (Path(__file__).resolve().parents[1] / "infra" / "cloudformation" / "artifacts.yaml").read_text(encoding="utf-8")
        for property_name in ("BucketName:", "RepositoryName:", "TableName:"):
            self.assertIsNone(
                re.search(rf"^      {property_name}", text, flags=re.M),
                f"fixed {property_name} prevents a clean retry after rollback",
            )

    def test_deploy_reuses_existing_immutable_image_tags(self):
        text = (Path(__file__).resolve().parents[1] / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("aws ecr describe-images", text)
        self.assertIn("Reusing existing $target image", text)
        self.assertIn("docker push --quiet \"$repo:$image_tag\"", text)

    def test_every_role_that_uses_the_encrypted_state_table_may_use_its_key_through_dynamodb(self):
        # The state table is encrypted with the artifact key: without these two statements every ledger
        # read or write fails with AccessDeniedException (found on the Fargate task role).
        for name in ("service.yaml", "serverless.yaml"):
            text = (Path(__file__).resolve().parents[1] / "infra" / "cloudformation" / name).read_text(encoding="utf-8")
            self.assertIn("dynamodb:UpdateItem", text, name)
            decrypt = text.split("Action: kms:Decrypt", 1)[1].split("- Effect", 1)[0]
            self.assertIn("dynamodb.${AWS::Region}.amazonaws.com", decrypt, f"{name}: kms:Decrypt via DynamoDB")
            generate = text.split("Action: kms:GenerateDataKey", 1)[1].split("\n\n", 1)[0]
            self.assertIn("dynamodb.${AWS::Region}.amazonaws.com", generate, f"{name}: kms:GenerateDataKey via DynamoDB")

    def test_lock_files_hold_no_local_paths(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "requirements").glob("*.txt")) + [root / "requirements.txt"]:
            text = path.read_text(encoding="utf-8")
            for marker in ("/tmp/", "scratchpad", "claude-0", "/home/"):
                self.assertNotIn(marker, text, f"{path.name} contains a local path ({marker})")

    def test_the_budget_rule_still_exists(self):
        text = (Path(__file__).resolve().parents[1] / "infra" / "cloudformation" / "monitoring.yaml").read_text(encoding="utf-8")
        self.assertIn("MonthlyBudgetRequiresEmail", rules_block(text))
        self.assertIn("BudgetEmail", rules_block(text))


if __name__ == "__main__":
    unittest.main()


class StateMachineNameTests(unittest.TestCase):
    def test_the_two_pipelines_never_share_a_state_machine_name(self):
        # CloudFormation creates the new pipeline before it deletes the old one when a stack switches backends; a shared
        # fixed name fails with "State machine already exists" (deploy runs 23 to 27).
        names = {}
        for template in ("pipeline.yaml", "pipeline-ecs.yaml"):
            text = (Path(__file__).resolve().parents[1] / "infra" / "cloudformation" / template).read_text(encoding="utf-8")
            match = re.search(r"StateMachineName: !Sub \"(.+?)\"", text)
            self.assertIsNotNone(match, template)
            names[template] = match.group(1)
        self.assertEqual(len(set(names.values())), 2, names)

    def test_no_resource_that_both_backends_create_has_the_same_fixed_name(self):
        root = Path(__file__).resolve().parents[1] / "infra" / "cloudformation"
        fixed = {}
        for template in ("pipeline.yaml", "pipeline-ecs.yaml", "mlflow-ecs.yaml", "evidently-ecs.yaml", "ecs-platform.yaml", "mlops.yaml"):
            text = (root / template).read_text(encoding="utf-8")
            for key in ("StateMachineName", "ClusterName", "ServiceName", "AlarmName", "LogGroupName", "Family"):
                for value in re.findall(rf"^\s+{key}: !Sub \"(.+?)\"", text, flags=re.M):
                    fixed.setdefault((key, value), []).append(template)
        clashes = {k: v for k, v in fixed.items() if len(v) > 1}
        self.assertEqual(clashes, {})
