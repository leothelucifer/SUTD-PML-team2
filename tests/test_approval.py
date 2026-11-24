"""Approve, reject and promote from GitHub: the approval script, the one-off task runner and the workflows."""

import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aws_approval as approval  # noqa: E402
import aws_run_task as run_task  # noqa: E402


class FakeStepFunctions:
    def __init__(self, waiting=("run-1",)):
        self.waiting = list(waiting)
        self.sent = []

    def list_executions(self, stateMachineArn, statusFilter):
        return {"executions": [{"name": n, "executionArn": f"arn:exec:{n}"} for n in self.waiting + ["run-busy"]]}

    def get_execution_history(self, executionArn, reverseOrder, maxResults):
        name = executionArn.rsplit(":", 1)[-1]
        if name == "run-busy":      # still training: nothing is waiting
            return {"events": [{"type": "TaskStateEntered"}]}
        params = {"TopicArn": "t", "MessageAttributes": {"token": {"DataType": "String", "StringValue": f"token-{name}"}}}
        return {"events": [{"type": "TaskScheduled", "taskScheduledEventDetails": {
            "resource": "publish.waitForTaskToken", "parameters": json.dumps(params)}}, {"type": "TaskStateEntered"}]}

    def send_task_success(self, taskToken, output):
        self.sent.append(("success", taskToken, json.loads(output)))

    def send_task_failure(self, taskToken, error, cause):
        self.sent.append(("failure", taskToken, error, cause))


class ApprovalTests(unittest.TestCase):
    def test_approve_answers_the_only_waiting_run_and_records_who(self):
        sfn = FakeStepFunctions()
        name = approval.decide(sfn, "arn:sm", "approve", "alice")
        self.assertEqual(name, "run-1")
        self.assertEqual(sfn.sent, [("success", "token-run-1", {"approved": True, "by": "alice"})])

    def test_reject_ends_the_run_with_the_reason_and_the_name(self):
        sfn = FakeStepFunctions()
        approval.decide(sfn, "arn:sm", "reject", "bob", reason="calibration looks off")
        kind, token, error, cause = sfn.sent[0]
        self.assertEqual((kind, token, error), ("failure", "token-run-1", "Rejected"))
        self.assertIn("rejected by bob: calibration looks off", cause)

    def test_two_waiting_runs_need_a_name_and_the_wrong_name_is_refused(self):
        sfn = FakeStepFunctions(waiting=("run-1", "run-2"))
        with self.assertRaisesRegex(ValueError, "several executions"):
            approval.decide(sfn, "arn:sm", "approve", "alice")
        with self.assertRaisesRegex(ValueError, "no execution named"):
            approval.decide(sfn, "arn:sm", "approve", "alice", name="nope")
        approval.decide(sfn, "arn:sm", "approve", "alice", name="run-2")
        self.assertEqual(sfn.sent[0][1], "token-run-2")

    def test_nothing_waiting_or_no_name_or_a_bad_decision_is_refused_and_nothing_is_sent(self):
        sfn = FakeStepFunctions(waiting=())
        with self.assertRaisesRegex(ValueError, "no pipeline execution is waiting"):
            approval.decide(sfn, "arn:sm", "approve", "alice")
        with self.assertRaisesRegex(ValueError, "name of the person"):
            approval.decide(FakeStepFunctions(), "arn:sm", "approve", "  ")
        with self.assertRaisesRegex(ValueError, "approve or reject"):
            approval.decide(FakeStepFunctions(), "arn:sm", "maybe", "alice")
        self.assertEqual(sfn.sent, [])

    def test_both_pipelines_publish_the_token_as_a_message_attribute(self):
        for name in ("pipeline.yaml", "pipeline-ecs.yaml"):
            text = (ROOT / "infra" / "cloudformation" / name).read_text(encoding="utf-8")
            self.assertIn("StringValue.$: $$.Task.Token", text, name)


class FakeEcs:
    def __init__(self, exit_code=0):
        self.exit_code, self.calls = exit_code, []

    def register_task_definition(self, **kw):
        self.calls.append(("register", kw))
        return {"taskDefinition": {"taskDefinitionArn": "arn:td:1"}}

    def run_task(self, **kw):
        self.calls.append(("run", kw))
        return {"tasks": [{"taskArn": "arn:aws:ecs:r:1:task/cluster/abc123"}], "failures": []}

    def describe_tasks(self, cluster, tasks):
        self.calls.append(("describe", tasks))
        return {"tasks": [{"lastStatus": "STOPPED", "containers": [{"exitCode": self.exit_code}]}]}

    def stop_task(self, **kw):
        self.calls.append(("stop", kw))


class FakeSession:
    def __init__(self, ecs, outputs=None):
        outputs = {"EcsClusterArn": "arn:aws:ecs:r:1:cluster/c", "EcsSubnetIds": "subnet-a,subnet-b",
                   "EcsTaskSecurityGroupId": "sg-1", "EcsExecutionRoleArn": "arn:exec", "EcsTaskRoleArn": "arn:task",
                   "EcsApproverTaskRoleArn": "arn:approver", "EcsLogGroupName": "/ecs/x",
                   "TrainingRepositoryUri": "1.dkr.ecr.r.amazonaws.com/train"} if outputs is None else outputs
        cf = type("Cf", (), {"describe_stacks": lambda self, StackName: {"Stacks": [{"Outputs": [
            {"OutputKey": k, "OutputValue": v} for k, v in outputs.items()]}]}})()
        logs = type("Logs", (), {"get_log_events": lambda self, **kw: {"events": [{"message": "Promoted version 3\n"}]}})()
        self.clients = {"ecs": ecs, "cloudformation": cf, "logs": logs}

    def client(self, name):
        return self.clients[name]


class RunTaskTests(unittest.TestCase):
    def run_it(self, ecs, **kw):
        lines = []
        code = run_task.run(FakeSession(ecs), "sepsis-dev", role=kw.get("role", "approver"), image_tag="training-abc",
                            env=["HP_mode=promote", "HP_reviewed_by=alice"], family_suffix="approver", region="ap-southeast-2",
                            sleep=lambda _: None, out=lines.append)
        return code, lines

    def test_the_approver_task_runs_as_the_approver_role_with_the_given_settings(self):
        ecs = FakeEcs()
        code, lines = self.run_it(ecs)
        self.assertEqual(code, 0)
        register = next(kw for kind, kw in ecs.calls if kind == "register")
        self.assertEqual(register["taskRoleArn"], "arn:approver")            # not the training task role
        self.assertEqual(register["containerDefinitions"][0]["image"], "1.dkr.ecr.r.amazonaws.com/train:training-abc")
        self.assertEqual(register["containerDefinitions"][0]["entryPoint"], ["python", "-m", "sepsis_warning.ecs_job"])
        run = next(kw for kind, kw in ecs.calls if kind == "run")
        self.assertEqual(run["networkConfiguration"]["awsvpcConfiguration"]["subnets"], ["subnet-a", "subnet-b"])
        env = {e["name"]: e["value"] for e in run["overrides"]["containerOverrides"][0]["environment"]}
        self.assertEqual(env, {"HP_mode": "promote", "HP_reviewed_by": "alice"})
        self.assertIn("Promoted version 3", lines)                            # the task's log is shown

    def test_the_training_role_is_used_for_ordinary_steps(self):
        ecs = FakeEcs()
        self.run_it(ecs, role="task")
        self.assertEqual(next(kw for kind, kw in ecs.calls if kind == "register")["taskRoleArn"], "arn:task")

    def test_the_exit_code_of_the_container_is_returned(self):
        code, _ = self.run_it(FakeEcs(exit_code=1))
        self.assertEqual(code, 1)

    def test_a_stack_without_the_ecs_outputs_is_refused_with_the_reason(self):
        session = FakeSession(FakeEcs(), outputs={"TrainingRepositoryUri": "x"})
        with self.assertRaisesRegex(ValueError, "EcsClusterArn"):
            run_task.run(session, "s", role="task", image_tag="t", env=[], family_suffix="f", region="r")

    def test_env_pairs_must_look_like_name_equals_value(self):
        self.assertEqual(run_task.parse_env(["A=1", "B=x=y"]), [{"name": "A", "value": "1"}, {"name": "B", "value": "x=y"}])
        with self.assertRaises(ValueError):
            run_task.parse_env(["no-equals"])


class WorkflowTests(unittest.TestCase):
    def load(self, name):
        return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))

    def test_promote_and_approve_are_manual_use_environment_reviewers_and_keep_inputs_out_of_scripts(self):
        for name, job in (("promote.yml", "promote"), ("approve.yml", "approve")):
            workflow = self.load(name)
            self.assertEqual(set(workflow[True]), {"workflow_dispatch"}, name)
            self.assertEqual(workflow["jobs"][job]["environment"], "${{ inputs.environment }}", name)
            self.assertEqual(workflow["permissions"]["id-token"], "write", name)
            for step in workflow["jobs"][job]["steps"]:
                self.assertNotIn("${{ inputs.", step.get("run", ""), f"{name}: {step.get('name')}")
                self.assertNotIn("${{ github.actor", step.get("run", ""), f"{name}: {step.get('name')}")

    def test_the_names_on_the_record_come_from_github_not_from_typed_text(self):
        promote = (ROOT / ".github" / "workflows" / "promote.yml").read_text(encoding="utf-8")
        approve = (ROOT / ".github" / "workflows" / "approve.yml").read_text(encoding="utf-8")
        self.assertIn("REVIEWED_BY: ${{ github.actor }}", promote)
        self.assertIn("DECIDED_BY: ${{ github.actor }}", approve)
        self.assertNotIn("reviewed_by:", promote)       # there is no input to type a different name into
        self.assertNotIn("decided_by:", approve)

    def test_the_deploy_script_has_the_two_commands(self):
        text = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("promote-task) cmd_promote_task", text)
        self.assertIn("approve-run) cmd_approve_run", text)


if __name__ == "__main__":
    unittest.main()


class ExistingFamilyTests(unittest.TestCase):
    def test_a_stack_owned_task_definition_is_run_as_it_is_and_its_log_is_found(self):
        class Ecs(FakeEcs):
            def describe_task_definition(self, taskDefinition):
                self.calls.append(("describe_definition", taskDefinition))
                return {"taskDefinition": {"containerDefinitions": [{"name": "report", "logConfiguration": {"options": {
                    "awslogs-group": "/ecs/x-evidently", "awslogs-stream-prefix": "report"}}}]}}

        ecs, streams = Ecs(), []
        session = FakeSession(ecs)
        session.clients["logs"] = type("Logs", (), {"get_log_events": lambda self, **kw: streams.append(kw) or {"events": []}})()
        code = run_task.run(session, "sepsis-dev", role="task", image_tag="", env=["REPORT_DAYS=2"], family_suffix="",
                            region="ap-southeast-2", sleep=lambda _: None, out=lambda *_: None,
                            existing_family="sepsis-dev-evidently-report", container="report")
        self.assertEqual(code, 0)
        self.assertFalse(any(kind == "register" for kind, *_ in ecs.calls))          # nothing is registered
        run = next(item[1] for item in ecs.calls if item[0] == "run")
        self.assertEqual(run["taskDefinition"], "sepsis-dev-evidently-report")
        self.assertEqual(run["overrides"]["containerOverrides"][0], {"name": "report", "environment": [{"name": "REPORT_DAYS", "value": "2"}]})
        self.assertEqual(streams[0]["logGroupName"], "/ecs/x-evidently")
        self.assertEqual(streams[0]["logStreamName"], "report/report/abc123")
