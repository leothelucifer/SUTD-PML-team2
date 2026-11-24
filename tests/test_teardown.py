"""Teardown: finds retained resources in nested stacks, plans by default, deletes only when confirmed."""

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aws_teardown", ROOT / "scripts" / "aws_teardown.py")
teardown = importlib.util.module_from_spec(spec)
sys.modules["aws_teardown"] = teardown
spec.loader.exec_module(teardown)


class FakeCloudFormation:
    def __init__(self):
        self.deleted = []
        self.stacks = {
            "root": [
                {"ResourceType": "AWS::CloudFormation::Stack", "PhysicalResourceId": "nested-artifacts",
                 "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::IAM::Role", "PhysicalResourceId": "role", "ResourceStatus": "CREATE_COMPLETE"},
            ],
            "nested-artifacts": [
                {"ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": "art-bucket", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::KMS::Key", "PhysicalResourceId": "key-1", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::DynamoDB::Table", "PhysicalResourceId": "table", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::Logs::LogGroup", "PhysicalResourceId": "/aws/lambda/x", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::ECR::Repository", "PhysicalResourceId": "repo", "ResourceStatus": "CREATE_COMPLETE"},
                {"ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": "gone", "ResourceStatus": "DELETE_COMPLETE"},
            ],
        }

    def list_stack_resources(self, StackName, **_):
        return {"StackResourceSummaries": self.stacks[StackName]}

    def delete_stack(self, StackName):
        self.deleted.append(StackName)

    def get_waiter(self, _):
        return Namespace(wait=lambda **kw: None)


class FakeS3:
    def __init__(self):
        self.calls = []

    def get_paginator(self, _):
        return Namespace(paginate=lambda Bucket: iter([{
            "Versions": [{"Key": "a", "VersionId": "1"}, {"Key": "a", "VersionId": "2"}],
            "DeleteMarkers": [{"Key": "b", "VersionId": "3"}]}]))

    def delete_objects(self, Bucket, Delete):
        self.calls.append(("objects", Bucket, len(Delete["Objects"])))

    def delete_bucket(self, Bucket):
        self.calls.append(("bucket", Bucket))


class Recorder:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def __getattr__(self, method):
        return lambda **kw: self.log.append((self.name, method, kw))


class FakeSession:
    def __init__(self):
        self.cfn, self.s3, self.log = FakeCloudFormation(), FakeS3(), []

    def client(self, name):
        return {"cloudformation": self.cfn, "s3": self.s3}.get(name) or Recorder(name, self.log)


ARGS = dict(stack="sepsis-dev", region="ap-southeast-2", templates_bucket="sepsis-dev-1-r-cfn")


class TeardownTests(unittest.TestCase):
    def test_collects_retained_resources_from_nested_stacks(self):
        found = teardown.collect(FakeCloudFormation(), "root")
        self.assertEqual(found, [("AWS::S3::Bucket", "art-bucket"), ("AWS::ECR::Repository", "repo"),
                                 ("AWS::DynamoDB::Table", "table"), ("AWS::Logs::LogGroup", "/aws/lambda/x"),
                                 ("AWS::KMS::Key", "key-1")])

    def test_plan_only_deletes_nothing(self):
        session = FakeSession()
        session.cfn.stacks["sepsis-dev"] = session.cfn.stacks["root"]
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=False, confirm=""), session), 0)
        self.assertEqual((session.cfn.deleted, session.s3.calls, session.log), ([], [], []))

    def test_wrong_confirmation_deletes_nothing(self):
        session = FakeSession()
        session.cfn.stacks["sepsis-dev"] = session.cfn.stacks["root"]
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=True, confirm="sepsis-prod"), session), 2)
        self.assertEqual((session.cfn.deleted, session.s3.calls), ([], []))

    def test_confirmed_run_deletes_stack_then_leftovers(self):
        session = FakeSession()
        session.cfn.stacks["sepsis-dev"] = session.cfn.stacks["root"]
        self.assertEqual(teardown.run(Namespace(**ARGS, yes=True, confirm="sepsis-dev"), session), 0)
        self.assertEqual(session.cfn.deleted, ["sepsis-dev"])
        self.assertIn(("objects", "art-bucket", 3), session.s3.calls)
        self.assertIn(("bucket", "art-bucket"), session.s3.calls)
        self.assertIn(("bucket", "sepsis-dev-1-r-cfn"), session.s3.calls)
        methods = [(n, m) for n, m, _ in session.log]
        self.assertIn(("ecr", "delete_repository"), methods)
        self.assertIn(("dynamodb", "delete_table"), methods)
        self.assertIn(("logs", "delete_log_group"), methods)
        key = [kw for n, m, kw in session.log if m == "schedule_key_deletion"][0]
        self.assertEqual(key["PendingWindowInDays"], 7)
        self.assertNotIn(("bucket", "gone"), session.s3.calls)

    def test_missing_stack_is_not_an_error(self):
        session = FakeSession()

        def missing(StackName, **_):
            raise RuntimeError("An error occurred (ValidationError): Stack with id x does not exist")

        session.cfn.list_stack_resources = missing
        with mock.patch.object(teardown, "leftovers", lambda *args: []):
            self.assertEqual(teardown.run(Namespace(**ARGS, yes=True, confirm="sepsis-dev"), session), 0)
        self.assertEqual((session.cfn.deleted, session.s3.calls), ([], []))

    def test_leftovers_of_a_failed_first_create_are_found_and_removed(self):
        session = FakeSession()

        def missing(StackName, **_):
            raise RuntimeError("Stack with id x does not exist")

        session.cfn.list_stack_resources = missing
        sts = Namespace(get_caller_identity=lambda: {"Account": "123456789012"})

        def absent(**kwargs):
            raise RuntimeError("NotFound")

        session.s3.head_bucket = lambda Bucket: None if Bucket.endswith("-artifacts") else absent()
        recorder = session.client
        tagged = Namespace(
            get_paginator=lambda _: Namespace(paginate=lambda: [{"Keys": [{"KeyId": "key-1"}, {"KeyId": "key-2"}]}]),
            list_resource_tags=lambda KeyId: {"Tags": [{"TagKey": "Name", "TagValue": "sepsis-dev-artifacts-key" if KeyId == "key-1" else "other"}]},
            describe_key=lambda KeyId: {"KeyMetadata": {"KeyState": "Enabled"}},
            schedule_key_deletion=lambda **kw: session.log.append(("kms", "schedule_key_deletion", kw)))
        ecr = Namespace(describe_repositories=lambda repositoryNames: None if repositoryNames[0].endswith("/training") else absent(),
                        delete_repository=lambda **kw: session.log.append(("ecr", "delete_repository", kw)))
        dynamodb = Namespace(describe_table=lambda TableName: absent(), update_table=lambda **kw: None, delete_table=lambda **kw: None)
        logs = Namespace(describe_log_groups=lambda logGroupNamePrefix: {"logGroups": []}, delete_log_group=lambda **kw: None)
        session.client = lambda name: {"sts": sts, "kms": tagged, "ecr": ecr, "dynamodb": dynamodb, "logs": logs}.get(name) or recorder(name)

        args = dict(stack="sepsis-dev", region="ap-southeast-2", templates_bucket=None)
        self.assertEqual(teardown.run(Namespace(**args, yes=False, confirm=""), session), 0)
        self.assertEqual(session.s3.calls, [])  # plan only
        self.assertEqual(teardown.run(Namespace(**args, yes=True, confirm="sepsis-dev"), session), 0)
        self.assertEqual(session.cfn.deleted, [])  # there is no stack to delete
        self.assertIn(("bucket", "sepsis-dev-123456789012-ap-southeast-2-artifacts"), session.s3.calls)
        self.assertNotIn(("bucket", "sepsis-dev-123456789012-ap-southeast-2-logs"), session.s3.calls)
        removed = [(m, kw) for n, m, kw in session.log]
        self.assertIn(("delete_repository", {"repositoryName": "sepsis-dev/training", "force": True}), removed)
        self.assertEqual([kw["KeyId"] for m, kw in removed if m == "schedule_key_deletion"], ["key-1"])

    def test_validation_failures_come_from_describe_events(self):
        spec = importlib.util.spec_from_file_location("aws_stack_failures", ROOT / "scripts" / "aws_stack_failures.py")
        failures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(failures)
        cfn = Namespace(describe_events=lambda StackName, Filters: {"OperationEvents": [
            {"ValidationName": "AWS::EarlyValidation::ResourceExistenceCheck", "ValidationPath": "/Resources/ArtifactBucket",
             "ValidationStatusReason": "bucket already exists", "ResourceType": "AWS::S3::Bucket"}]},
            describe_stack_events=lambda StackName, **kw: {"StackEvents": []})
        text = failures.report(failures.failures(cfn, "arn:aws:cloudformation:r:1:stack/n/x"))
        self.assertIn("bucket already exists", text)
        self.assertIn("ArtifactBucket", text)
    def test_change_set_failures_show_the_name_conflicts(self):
        spec = importlib.util.spec_from_file_location("aws_stack_failures", ROOT / "scripts" / "aws_stack_failures.py")
        failures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(failures)
        cfn = Namespace(
            describe_change_set=lambda StackName, ChangeSetName: {"StatusReason": "Validation failed with 1 error(s)"},
            describe_events=lambda StackName, ChangeSetName, Filters: {"OperationEvents": [
                {"ValidationName": "NAME_CONFLICT_VALIDATION", "ValidationPath": "/Resources/StateTable",
                 "ValidationStatusReason": "Resource of type 'AWS::DynamoDB::Table' with identifier 'x' already exists.",
                 "ResourceType": "AWS::DynamoDB::Table"}]})
        text = failures.report(failures.change_set_failures(cfn, "sepsis-dev", "cs-1"))
        self.assertIn("already exists", text)
        self.assertIn("StateTable", text)

    def test_deploy_validates_before_it_creates_anything(self):
        script = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("validate-template", script)
        self.assertIn("--no-execute-changeset", script)
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8"))
        names = [step.get("name", "") for step in workflow["jobs"]["deploy"]["steps"]]
        validate = next(i for i, n in enumerate(names) if n.startswith("Validate the templates"))
        stack = names.index("Stack and images")
        self.assertLess(validate, stack)

    def test_script_and_workflow_require_confirmation(self):
        script = (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8")
        self.assertIn("CONFIRM_DESTROY", script)
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "teardown.yml").read_text(encoding="utf-8"))
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["scope"]["default"], "plan only")
        self.assertEqual(inputs["scope"]["options"], ["plan only", "stack only", "everything"])
        self.assertEqual(workflow["jobs"]["teardown"]["environment"], "${{ inputs.environment }}")
        self.assertEqual(workflow["permissions"]["id-token"], "write")


if __name__ == "__main__":
    unittest.main()


class StackFailureTests(unittest.TestCase):
    def test_reports_nested_failures_and_skips_cancellations(self):
        spec = importlib.util.spec_from_file_location("aws_stack_failures", ROOT / "scripts" / "aws_stack_failures.py")
        failures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(failures)

        nested = "arn:aws:cloudformation:ap-southeast-2:1:stack/sepsis-dev-Pipeline-X/abc"
        events = {
            "sepsis-dev": [
                {"StackName": "sepsis-dev", "LogicalResourceId": "Pipeline", "ResourceType": "AWS::CloudFormation::Stack",
                 "PhysicalResourceId": nested, "ResourceStatus": "CREATE_FAILED",
                 "ResourceStatusReason": "Embedded stack was not successfully created"},
                {"StackName": "sepsis-dev", "LogicalResourceId": "Artifacts", "ResourceType": "AWS::CloudFormation::Stack",
                 "PhysicalResourceId": "", "ResourceStatus": "CREATE_FAILED", "ResourceStatusReason": "Resource creation cancelled"},
            ],
            nested: [
                {"StackName": "sepsis-dev-Pipeline-X", "LogicalResourceId": "TrainingPipeline", "ResourceType": "AWS::StepFunctions::StateMachine",
                 "PhysicalResourceId": "", "ResourceStatus": "CREATE_FAILED", "ResourceStatusReason": "'null' values are not allowed"},
            ],
        }
        cfn = Namespace(describe_stack_events=lambda StackName, **kw: {"StackEvents": events[StackName]})
        items = failures.failures(cfn, "sepsis-dev")
        text = failures.report(items)
        self.assertIn("'null' values are not allowed", text)
        self.assertNotIn("cancelled", text)
        self.assertIn("TrainingPipeline", text)


class StaleFailureTests(unittest.TestCase):
    """An earlier failed attempt must not be reported as the cause of the latest one (deploy runs 23 to 27)."""

    def load(self):
        spec = importlib.util.spec_from_file_location("aws_stack_failures", ROOT / "scripts" / "aws_stack_failures.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_only_the_latest_operation_is_reported(self):
        import datetime

        failures = self.load()
        t = lambda hour, minute=0: datetime.datetime(2026, 11, 24, hour, minute, tzinfo=datetime.UTC)  # noqa: E731
        old_nested = "arn:aws:cloudformation:ap-southeast-2:1:stack/sepsis-dev-EcsPlatform-OLD/aaa"
        new_nested = "arn:aws:cloudformation:ap-southeast-2:1:stack/sepsis-dev-PipelineEcs-NEW/bbb"
        events = {
            "sepsis-dev": [   # newest first, as DescribeStackEvents returns them
                {"StackName": "sepsis-dev", "LogicalResourceId": "PipelineEcs", "ResourceType": "AWS::CloudFormation::Stack",
                 "PhysicalResourceId": new_nested, "ResourceStatus": "CREATE_FAILED", "Timestamp": t(15, 25),
                 "ResourceStatusReason": "Embedded stack was not successfully created"},
                {"StackName": "sepsis-dev", "LogicalResourceId": "sepsis-dev", "ResourceType": "AWS::CloudFormation::Stack",
                 "PhysicalResourceId": "arn:aws:cloudformation:ap-southeast-2:1:stack/sepsis-dev/root", "ResourceStatus": "UPDATE_IN_PROGRESS",
                 "Timestamp": t(15, 12), "ResourceStatusReason": "User Initiated"},
                {"StackName": "sepsis-dev", "LogicalResourceId": "EcsPlatform", "ResourceType": "AWS::CloudFormation::Stack",
                 "PhysicalResourceId": old_nested, "ResourceStatus": "CREATE_FAILED", "Timestamp": t(9, 13),
                 "ResourceStatusReason": "Embedded stack was not successfully created"},
            ],
            new_nested: [
                {"StackName": "sepsis-dev-PipelineEcs-NEW", "LogicalResourceId": "TrainingPipeline", "ResourceType": "AWS::StepFunctions::StateMachine",
                 "PhysicalResourceId": "", "ResourceStatus": "CREATE_FAILED", "Timestamp": t(15, 24),
                 "ResourceStatusReason": "State machine already exists"},
            ],
            old_nested: [
                {"StackName": "sepsis-dev-EcsPlatform-OLD", "LogicalResourceId": "Cluster", "ResourceType": "AWS::ECS::Cluster",
                 "PhysicalResourceId": "", "ResourceStatus": "CREATE_FAILED", "Timestamp": t(9, 13),
                 "ResourceStatusReason": "Unable to assume the service linked role"},
            ],
        }
        cfn = Namespace(describe_stack_events=lambda StackName, **kw: {"StackEvents": events[StackName]})
        text = failures.report(failures.failures(cfn, "sepsis-dev"))
        self.assertIn("State machine already exists", text)
        self.assertNotIn("service linked role", text)               # the old attempt, long fixed
        self.assertNotIn("EcsPlatform-OLD", text)

    def test_events_without_timestamps_are_all_kept(self):
        failures = self.load()
        events = [{"StackName": "s", "LogicalResourceId": "X", "ResourceType": "AWS::S3::Bucket", "PhysicalResourceId": "",
                   "ResourceStatus": "CREATE_FAILED", "ResourceStatusReason": "boom"}]
        cfn = Namespace(describe_stack_events=lambda StackName, **kw: {"StackEvents": events})
        self.assertIn("boom", failures.report(failures.failures(cfn, "s")))
