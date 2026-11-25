"""One deploy button decides its stages from what changed (scripts/deploy_plan.py)."""

import io
import json
import unittest
from datetime import UTC, datetime
from pathlib import Path

from scripts import ci_changes
from scripts import deploy_plan as dp

RULES = ci_changes.load_rules()
HEAD = "h" * 40
OLD = "o" * 40


def state(stack=OLD, retrain=OLD, serve=OLD, model=1):
    stages = {k: v for k, v in (("stack", stack), ("retrain", retrain), ("serve", serve)) if v}
    return {"version": 1, "stages": stages, **({"model_version": model} if model else {})}


def decide(files=None, **kwargs):
    defaults = dict(state=state(), head=HEAD, rules=RULES, mode="auto", retrain="auto", stack_exists=True,
                    model_versions=[1], pipeline_running=False)
    defaults.update(kwargs)
    changed = files if files is not None else []
    return dp.decide(files_between=lambda base, head: changed, **defaults)


def flags(result):
    return {k for k, v in result["flags"].items() if v}


class AutoModeTests(unittest.TestCase):
    def test_docs_only_does_nothing(self):
        result = decide(["docs/a.md"])
        self.assertEqual(flags(result), set())

    def test_serving_code_rebuilds_and_serves_without_touching_the_stack_or_training(self):
        result = decide(["src/sepsis_warning/api.py"])
        self.assertEqual(flags(result), {"serve", "images", "validate", "verify", "any"})

    def test_training_code_starts_a_retrain(self):
        result = decide(["src/sepsis_warning/tuning.py"])
        self.assertTrue(result["flags"]["retrain"])
        self.assertFalse(result["flags"]["serve"])
        self.assertFalse(result["flags"]["stack"])

    def test_feature_code_retrains_and_serves(self):
        result = decide(["src/sepsis_warning/features.py"])
        self.assertTrue(result["flags"]["retrain"])
        self.assertTrue(result["flags"]["serve"])

    def test_infra_updates_the_stack_and_waits_for_the_tools(self):
        result = decide(["infra/cloudformation/mlflow-ecs.yaml"])
        self.assertTrue(result["flags"]["stack"])
        self.assertTrue(result["flags"]["ready"])
        self.assertFalse(result["flags"]["serve"])

    def test_dependencies_rebuild_everything_and_serve_again(self):
        result = decide(["requirements/lambda.txt"])
        self.assertTrue(result["flags"]["stack"])
        self.assertTrue(result["flags"]["serve"])

    def test_a_newer_approved_model_is_served_even_with_no_code_change(self):
        result = decide([], model_versions=[1, 2], state=state(model=1))
        self.assertTrue(result["flags"]["serve"])
        self.assertIn("v2", result["reasons"]["serve"])

    def test_the_served_model_is_not_served_again(self):
        self.assertFalse(decide([], model_versions=[2], state=state(model=2))["flags"]["serve"])


class FreshEnvironmentTests(unittest.TestCase):
    def test_a_new_account_creates_only_the_stack_first(self):
        result = decide(state=None, stack_exists=False, model_versions=[])
        self.assertEqual({"stack", "ready"} & flags(result), {"stack", "ready"})
        self.assertFalse(result["flags"]["retrain"])
        self.assertFalse(result["flags"]["serve"])
        self.assertTrue(any("Training needs the stack" in n for n in result["notes"]))

    def test_the_second_click_starts_training(self):
        result = decide(state=state(retrain=None, serve=None, model=None), model_versions=[])
        self.assertTrue(result["flags"]["retrain"])
        self.assertFalse(result["flags"]["serve"])
        self.assertTrue(any("no approved model" in n for n in result["notes"]))

    def test_the_click_after_approval_serves_the_model(self):
        result = decide(state=state(serve=None, model=None), model_versions=[1])
        self.assertTrue(result["flags"]["serve"])
        self.assertFalse(result["flags"]["retrain"])

    def test_an_environment_that_already_has_a_model_does_not_train_from_scratch(self):
        result = decide(state=state(retrain=None, model=None), model_versions=[3])
        self.assertFalse(result["flags"]["retrain"])
        self.assertIn("already published", result["reasons"]["retrain"])


class OverrideTests(unittest.TestCase):
    def test_stack_only_runs_only_the_stack(self):
        result = decide(["src/sepsis_warning/features.py"], mode="stack and images only")
        self.assertEqual({s for s in dp.STAGES if result["flags"][s]}, {"stack"})

    def test_full_deploy_runs_stack_and_serve_but_retrains_only_without_a_model(self):
        with_model = decide([], mode="full deploy")
        self.assertTrue(with_model["flags"]["stack"] and with_model["flags"]["serve"])
        self.assertFalse(with_model["flags"]["retrain"])
        without = decide([], mode="full deploy", model_versions=[], state=state(retrain=None, model=None))
        self.assertFalse(without["flags"]["serve"])
        self.assertTrue(without["flags"]["retrain"])

    def test_retrain_yes_forces_it_and_no_prevents_it(self):
        self.assertTrue(decide(["docs/a.md"], retrain="yes")["flags"]["retrain"])
        self.assertFalse(decide(["src/sepsis_warning/tuning.py"], retrain="no")["flags"]["retrain"])

    def test_a_running_pipeline_is_never_started_twice(self):
        result = decide(["src/sepsis_warning/tuning.py"], pipeline_running=True)
        self.assertFalse(result["flags"]["retrain"])
        self.assertTrue(any("in progress" in n for n in result["notes"]))
        self.assertFalse(decide([], retrain="yes", pipeline_running=True)["flags"]["retrain"])

    def test_bad_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            decide(mode="everything")
        with self.assertRaises(ValueError):
            decide(retrain="maybe")

    def test_an_unreachable_base_commit_assumes_changed(self):
        result = dp.decide(state=state(), head=HEAD, rules=RULES, mode="auto", retrain="auto", stack_exists=True,
                           model_versions=[1], pipeline_running=False, files_between=lambda base, head: None)
        self.assertTrue(result["flags"]["stack"])
        self.assertIn("assuming changed", result["reasons"]["stack"])


class OutputTests(unittest.TestCase):
    def test_a_dry_run_executes_nothing_but_says_what_it_would_do(self):
        import tempfile
        result = decide(["src/sepsis_warning/api.py"])
        with tempfile.TemporaryDirectory() as temp:
            out, summary = Path(temp) / "out", Path(temp) / "summary"
            dp.write_outputs(result, True, str(out), str(summary))
            lines = out.read_text().splitlines()
            self.assertIn("serve=false", lines)
            self.assertIn("dry run", summary.read_text())
            self.assertIn("| serve | yes |", summary.read_text())

    def test_outputs_for_a_real_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            dp.write_outputs(decide(["src/sepsis_warning/api.py"]), False, str(out), None)
            lines = out.read_text().splitlines()
            self.assertIn("serve=true", lines)
            self.assertIn("stack=false", lines)
            self.assertIn("stages=serve", lines)

    def test_nothing_to_do_is_said_plainly(self):
        self.assertIn("Nothing to do", dp.markdown(decide([]), False))


class StateTests(unittest.TestCase):
    NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)

    def test_record_sets_only_the_stages_that_ran(self):
        new = dp.record(state(), HEAD, ["serve"], 2, "alice", "42", self.NOW)
        self.assertEqual(new["stages"], {"stack": OLD, "retrain": OLD, "serve": HEAD})
        self.assertEqual(new["model_version"], 2)
        self.assertEqual((new["by"], new["run"]), ("alice", "42"))

    def test_the_model_version_changes_only_when_serving_ran(self):
        new = dp.record(state(model=1), HEAD, ["stack"], 5, "a", "1", self.NOW)
        self.assertEqual(new["model_version"], 1)

    def test_a_first_record_starts_from_nothing(self):
        new = dp.record(None, HEAD, ["stack"], None, "a", "1", self.NOW)
        self.assertEqual(new["stages"], {"stack": HEAD})

    def test_an_unknown_stage_is_refused(self):
        with self.assertRaises(ValueError):
            dp.record(None, HEAD, ["images"], None, "a", "1")

    def test_state_round_trips_and_a_missing_file_means_no_state(self):
        class Body:
            def __init__(self, data): self.data = data
            def read(self): return self.data

        class S3:
            def __init__(self): self.store = {}
            def put_object(self, Bucket, Key, Body, ContentType): self.store[(Bucket, Key)] = Body
            def get_object(self, Bucket, Key):
                if (Bucket, Key) not in self.store:
                    error = Exception("no key")
                    error.response = {"Error": {"Code": "NoSuchKey"}}
                    raise error
                return {"Body": io.BytesIO(self.store[(Bucket, Key)])}

        s3 = S3()
        self.assertIsNone(dp.read_state(s3, "b", "dev"))
        dp.write_state(s3, "b", "dev", state())
        self.assertEqual(dp.read_state(s3, "b", "dev")["stages"]["stack"], OLD)
        self.assertEqual(json.loads(s3.store[("b", "deployments/dev/state.json")])["version"], 1)


if __name__ == "__main__":
    unittest.main()
