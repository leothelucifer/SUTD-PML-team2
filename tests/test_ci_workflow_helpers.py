"""Path and AWS-target guards used by GitHub Actions."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.ci_targets import ROOT, TARGETS
from scripts.resolve_aws_target import resolve_target


class CiTargetTests(unittest.TestCase):
    def test_every_configured_type_and_coverage_target_exists(self):
        for gate, targets in TARGETS.items():
            with self.subTest(gate=gate):
                self.assertTrue(all((ROOT / path).is_file() for path in targets))


class AwsTargetTests(unittest.TestCase):
    def test_explicit_role_precedes_account_fallback(self):
        region, role = resolve_target(
            "dev", "ap-southeast-2", "ap-southeast-1", "not-used",
            "arn:aws:iam::123456789012:role/custom",
        )
        self.assertEqual(region, "ap-southeast-2")
        self.assertEqual(role, "arn:aws:iam::123456789012:role/custom")

    def test_environment_default_uses_the_account_and_resolved_region(self):
        region, role = resolve_target("staging", "environment default", "ap-southeast-1", "123456789012", "")
        self.assertEqual(region, "ap-southeast-1")
        self.assertEqual(role, "arn:aws:iam::123456789012:role/github-deploy-staging-ap-southeast-1")

    def test_invalid_target_or_missing_identity_is_rejected(self):
        for args in (
            ("dev", "us-east-1", "ap-southeast-1", "123456789012", ""),
            ("dev", "environment default", "", "123456789012", ""),
            ("dev", "ap-southeast-1", "", "", ""),
            ("dev", "ap-southeast-1", "", "not-an-account", ""),
            ("dev", "ap-southeast-1", "", "", "arn:bad\nrole"),
            ("other", "ap-southeast-1", "", "123456789012", ""),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_target(*args)

    def test_action_helper_writes_resolved_outputs_for_github(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "github-output"
            environment = os.environ | {
                "ENVIRONMENT": "dev",
                "REGION_INPUT": "environment default",
                "DEFAULT_REGION": "ap-southeast-2",
                "ACCOUNT_ID": "123456789012",
                "ROLE_ARN_VAR": "",
                "GITHUB_OUTPUT": str(output),
            }
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/resolve_aws_target.py")],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                output.read_text(),
                "region=ap-southeast-2\nrole=arn:aws:iam::123456789012:role/github-deploy-dev-ap-southeast-2\n",
            )


if __name__ == "__main__":
    unittest.main()
