"""A stack still updating from an earlier (cancelled) run is settled before a new change set is made (deploy run 29)."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")


def run(statuses: list[str]):
    """statuses: what describe-stacks returns on the 1st, 2nd, ... call (the last repeats)."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        log, counter = temp / "calls.log", temp / "count"
        counter.write_text("0")
        fake = temp / "aws"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "echo \"$@\" >> \"$CALLS\"\n"
            "if [ \"$1 $2\" = 'cloudformation describe-stacks' ]; then\n"
            "  n=$(cat \"$COUNT\"); echo $((n+1)) > \"$COUNT\"\n"
            "  IFS=, read -ra S <<< \"$STATUSES\"; i=$n; [ $i -ge ${#S[@]} ] && i=$((${#S[@]}-1)); echo \"${S[$i]}\"\n"
            "fi\n"
            "exit 0\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        function = SCRIPT[SCRIPT.index("settle_stack() {"):]
        function = function[:function.index("\n}\n") + 3]
        env = {**os.environ, "PATH": f"{temp}:{os.environ['PATH']}", "CALLS": str(log), "COUNT": str(counter),
               "STATUSES": ",".join(statuses)}
        result = subprocess.run(["bash", "-c", 'fail() { echo "FAIL: $*" >&2; exit 1; }\nSTACK=sepsis-dev\n' + function + "\nsettle_stack"],
                                env=env, capture_output=True, text=True)
        return result, log.read_text() if log.exists() else ""


class SettleStackTests(unittest.TestCase):
    def test_a_finished_stack_is_left_alone(self):
        result, calls = run(["UPDATE_COMPLETE"])
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("cancel-update-stack", calls)

    def test_a_missing_stack_is_left_alone(self):
        result, calls = run([""])
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("cancel-update-stack", calls)

    def test_an_update_in_progress_is_cancelled_and_waited_for(self):
        result, calls = run(["UPDATE_IN_PROGRESS", "UPDATE_ROLLBACK_COMPLETE"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cancel-update-stack --stack-name sepsis-dev", calls)
        self.assertIn("wait stack-rollback-complete --stack-name sepsis-dev", calls)

    def test_a_rollback_already_running_is_waited_for_not_cancelled(self):
        result, calls = run(["UPDATE_ROLLBACK_IN_PROGRESS", "UPDATE_ROLLBACK_COMPLETE"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("cancel-update-stack", calls)
        self.assertIn("wait stack-rollback-complete", calls)

    def test_a_stack_that_stays_busy_stops_the_deploy_with_a_reason(self):
        result, _ = run(["UPDATE_IN_PROGRESS", "UPDATE_IN_PROGRESS"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("UPDATE_IN_PROGRESS", result.stderr)

    def test_a_failed_rollback_asks_for_the_console(self):
        result, _ = run(["UPDATE_ROLLBACK_FAILED"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("UPDATE_ROLLBACK_FAILED", result.stderr)


if __name__ == "__main__":
    unittest.main()
