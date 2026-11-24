"""A new account has no ECS service-linked role: the deploy creates it first (deploy run 22 failed on this)."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")


def run_function(*, role_exists: bool, create_output: str = "", create_ok: bool = True):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        log = temp / "calls.log"
        fake = temp / "aws"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "echo \"$@\" >> \"$CALLS\"\n"
            "if [ \"$1 $2\" = 'iam get-role' ]; then [ \"$ROLE_EXISTS\" = 1 ] && exit 0 || exit 254; fi\n"
            "if [ \"$1 $2\" = 'iam create-service-linked-role' ]; then echo \"$CREATE_OUTPUT\"; [ \"$CREATE_OK\" = 1 ] && exit 0 || exit 254; fi\n"
            "exit 0\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        function = SCRIPT[SCRIPT.index("ensure_service_linked_role() {"):]
        function = function[:function.index("\n}\n") + 3]
        env = {**os.environ, "PATH": f"{temp}:{os.environ['PATH']}", "CALLS": str(log), "ROLE_EXISTS": "1" if role_exists else "0",
               "CREATE_OUTPUT": create_output, "CREATE_OK": "1" if create_ok else "0"}
        result = subprocess.run(["bash", "-c", 'fail() { echo "FAIL: $*" >&2; exit 1; }\n' + function +
                                 "\nensure_service_linked_role ecs.amazonaws.com AWSServiceRoleForECS"],
                                env=env, capture_output=True, text=True)
        return result, log.read_text() if log.exists() else ""


class ServiceLinkedRoleTests(unittest.TestCase):
    def test_an_existing_role_is_left_alone(self):
        result, calls = run_function(role_exists=True)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("create-service-linked-role", calls)
        self.assertIn("iam wait role-exists --role-name AWSServiceRoleForECS", calls)

    def test_a_missing_role_is_created_with_the_documented_command(self):
        result, calls = run_function(role_exists=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("iam get-role --role-name AWSServiceRoleForECS", calls)
        self.assertIn("iam create-service-linked-role --aws-service-name ecs.amazonaws.com", calls)
        self.assertIn("iam wait role-exists --role-name AWSServiceRoleForECS", calls)

    def test_a_role_created_by_someone_else_in_the_meantime_is_not_an_error(self):
        result, _ = run_function(role_exists=False, create_ok=False,
                                 create_output="InvalidInput: Service role name AWSServiceRoleForECS has been taken in this account")
        self.assertEqual(result.returncode, 0)

    def test_a_denied_creation_stops_with_what_to_ask_an_administrator(self):
        result, _ = run_function(role_exists=False, create_ok=False, create_output="AccessDenied: explicit deny in a service control policy")
        self.assertEqual(result.returncode, 1)
        self.assertIn("aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com", result.stderr)
        self.assertIn("service control policy", result.stderr)

    def test_the_stack_command_ensures_the_roles_before_it_deploys_and_only_for_ecs_and_rds(self):
        stack = SCRIPT[SCRIPT.index("cmd_stack() {"):SCRIPT.index("\n}\n", SCRIPT.index("cmd_stack() {"))]
        self.assertLess(stack.index("ensure_service_linked_role ecs.amazonaws.com AWSServiceRoleForECS"),
                        stack.index("aws cloudformation deploy"))
        self.assertIn("ensure_service_linked_role rds.amazonaws.com AWSServiceRoleForRDS", stack)
        self.assertIn("TrainingBackend=ecs|MlflowBackend=ecs", stack)
        self.assertIn('SERVING_MODE" = fargate', stack)


if __name__ == "__main__":
    unittest.main()
