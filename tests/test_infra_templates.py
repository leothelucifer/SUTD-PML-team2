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

    def test_the_budget_rule_still_exists(self):
        text = (Path(__file__).resolve().parents[1] / "infra" / "cloudformation" / "monitoring.yaml").read_text(encoding="utf-8")
        self.assertIn("MonthlyBudgetRequiresEmail", rules_block(text))
        self.assertIn("BudgetEmail", rules_block(text))


if __name__ == "__main__":
    unittest.main()
