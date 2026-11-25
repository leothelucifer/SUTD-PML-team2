"""The deploy workflow lets you pick the environment and region, and uses no stored AWS keys."""

import unittest
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"


class DeployWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.text = PATH.read_text(encoding="utf-8")
        self.workflow = yaml.safe_load(self.text)
        self.inputs = self.workflow[True]["workflow_dispatch"]["inputs"]

    def test_environment_and_region_are_chosen_at_start(self):
        self.assertEqual(self.inputs["environment"]["options"], ["dev", "staging", "prod"])
        regions = self.inputs["region"]["options"]
        self.assertIn("ap-southeast-1", regions)
        self.assertIn("ap-southeast-2", regions)
        self.assertEqual(self.workflow["jobs"]["deploy"]["environment"], "${{ inputs.environment }}")

    def test_first_deploy_can_stop_before_publishing(self):
        steps = self.inputs["steps"]
        self.assertEqual(steps["default"], "stack and images only")
        gated = [s["name"] for s in self.workflow["jobs"]["deploy"]["steps"] if s.get("if") == "inputs.steps == 'full deploy'"]
        self.assertEqual(gated, ["Check that a model is ready to deploy (full deploy only)", "Publish the promoted model and serve it",
                                 "Smoke test the deployed API"])

    def test_verify_is_not_run_as_a_second_failure_after_publish_fails(self):
        steps = {step["name"]: step for step in self.workflow["jobs"]["deploy"]["steps"] if "name" in step}
        publish = steps["Publish the promoted model and serve it"]
        verify = steps["Verify every deployed component"]
        self.assertEqual(publish["id"], "publish")
        self.assertIn("steps.publish.outcome == 'success'", verify["if"])

    def test_credentials_come_from_oidc_only(self):
        self.assertEqual(self.workflow["permissions"]["id-token"], "write")
        self.assertNotIn("secrets.AWS", self.text)
        self.assertNotIn("aws-access-key-id", self.text)
        self.assertIn("steps.target.outputs.role", self.text)

    def test_role_name_matches_the_bootstrap_template(self):
        template = (PATH.parents[2] / "infra" / "bootstrap" / "github-oidc.yaml").read_text(encoding="utf-8")
        for environment in ("dev", "staging", "prod"):
            self.assertIn(f"github-deploy-{environment}-${{AWS::Region}}", template)
        self.assertIn("role/github-deploy-${ENVIRONMENT}-${region}", self.text)

    def test_explicit_role_takes_precedence_over_account_role(self):
        explicit = self.text.index('if [ -n "$ROLE_ARN_VAR" ]; then')
        account = self.text.index('elif [ -n "$ACCOUNT_ID" ]; then')
        self.assertLess(explicit, account)
        self.assertIn('role="$ROLE_ARN_VAR"', self.text)

    def test_trust_policy_accepts_the_id_based_subject(self):
        template = (PATH.parents[2] / "infra" / "bootstrap" / "github-oidc.yaml").read_text(encoding="utf-8")
        for environment in ("dev", "staging", "prod"):
            self.assertIn(f"repo:${{GitHubRepository}}:environment:{environment}", template)
            self.assertIn(f"repo:${{OwnerName}}@${{GitHubOwnerId}}/${{RepoName}}@${{GitHubRepositoryId}}:environment:{environment}", template)
        script = (PATH.parents[2] / "scripts" / "bootstrap_github_oidc.sh").read_text(encoding="utf-8")
        self.assertIn("GitHubOwnerId=", script)
        self.assertIn("GitHubRepositoryId=", script)

    def test_workflows_with_aws_access_install_only_hash_locked_packages(self):
        import re

        for name in ("deploy.yml", "pipeline.yml", "verify.yml", "teardown.yml"):
            workflow = PATH.parents[2] / ".github" / "workflows" / name
            for line in workflow.read_text(encoding="utf-8").splitlines():
                if "pip install" not in line or line.strip().startswith("#"):
                    continue
                locked = "--require-hashes" in line
                own_package = re.search(r"pip install --no-deps (-e )?\.", line)
                tooling = re.search(r"pip install (--upgrade )?(pip|uv|pip-audit)\b", line)
                self.assertTrue(locked or own_package or tooling, f"{workflow.name}: unpinned install: {line.strip()}")

    def test_mlflow_floor_matches_the_api_the_registry_uses(self):
        text = (PATH.parents[2] / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('tracking = ["mlflow>=3.0,<4"]', text)    # registry.py logs models with the MLflow 3 name= argument

    def test_untrusted_text_goes_through_env_not_inline(self):
        for step in self.workflow["jobs"]["deploy"]["steps"]:
            self.assertNotIn("inputs.model_uri", step.get("run", ""))


if __name__ == "__main__":
    unittest.main()
