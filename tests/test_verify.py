"""aws_verify: finds missing or unhealthy components without touching AWS."""

import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aws_verify", ROOT / "scripts" / "aws_verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def summary(type_, logical, physical, status="CREATE_COMPLETE"):
    return {"ResourceType": type_, "LogicalResourceId": logical, "PhysicalResourceId": physical, "ResourceStatus": status}


class FakeCloudFormation:
    def __init__(self, stacks):
        self.stacks = stacks

    def list_stack_resources(self, StackName, **_):
        return {"StackResourceSummaries": self.stacks[StackName]}


def statuses(result):
    return {component: status for status, component, _ in result.rows}


class VerifyTests(unittest.TestCase):
    def test_resources_follow_nested_stacks(self):
        cfn = FakeCloudFormation({
            "root": [summary("AWS::CloudFormation::Stack", "Artifacts", "arn:aws:cloudformation:r:1:stack/nested/x")],
            "arn:aws:cloudformation:r:1:stack/nested/x": [summary("AWS::S3::Bucket", "ArtifactBucket", "b")]})
        found = verify.resources(cfn, "root")
        self.assertEqual([i["logical"] for i in found], ["Artifacts", "ArtifactBucket"])

    def test_repositories_need_images_and_the_tag(self):
        items = [{"type": "AWS::ECR::Repository", "physical": name} for name in ("a", "b", "c")]
        images = {"a": [], "b": [{"imageTags": ["training-0ld0ld0ld0ld"]}], "c": [{"imageTags": ["training-abc123def456"]}]}
        ecr = Namespace(describe_images=lambda repositoryName: {"imageDetails": images[repositoryName]})
        result = verify.Result()
        verify.check_repositories(result, ecr, items, "abc123def456")
        self.assertEqual(statuses(result), {"ECR a": "FAIL", "ECR b": "FAIL", "ECR c": "PASS"})
        self.assertTrue(result.failed())

    def test_pending_email_is_a_warning_not_a_failure(self):
        items = [{"type": "AWS::SNS::Topic", "logical": "AlarmTopic", "physical": "arn:topic"},
                 {"type": "AWS::CloudWatch::Alarm", "logical": "A", "physical": "alarm-1"}]
        sns = Namespace(list_subscriptions_by_topic=lambda TopicArn: {"Subscriptions": [{"SubscriptionArn": "PendingConfirmation"}]})
        cloudwatch = Namespace(describe_alarms=lambda AlarmNames: {"MetricAlarms": [{"AlarmName": "alarm-1", "StateValue": "INSUFFICIENT_DATA"}]})
        result = verify.Result()
        verify.check_alerts(result, sns, cloudwatch, items)
        self.assertEqual(statuses(result), {"SNS AlarmTopic": "WARN", "CloudWatch alarms": "PASS"})
        self.assertFalse(result.failed())

    def test_a_firing_alarm_fails(self):
        items = [{"type": "AWS::CloudWatch::Alarm", "logical": "A", "physical": "alarm-1"}]
        cloudwatch = Namespace(describe_alarms=lambda AlarmNames: {"MetricAlarms": [{"AlarmName": "alarm-1", "StateValue": "ALARM"}]})
        result = verify.Result()
        verify.check_alerts(result, Namespace(), cloudwatch, items)
        self.assertEqual(statuses(result)["CloudWatch alarms"], "FAIL")

    def test_the_scoring_function_is_only_required_after_a_full_deploy(self):
        for expect, wanted in ((False, "WARN"), (True, "FAIL")):
            result = verify.Result()
            verify.check_function(result, Namespace(), [], {}, expect)
            self.assertEqual(statuses(result)["Lambda scoring function"], wanted)

    def test_a_healthy_function_and_signed_call_pass(self):
        items = [{"type": "AWS::Lambda::Function", "logical": "ScoringFunction", "physical": "sepsis-dev-scoring"}]
        client = Namespace(get_function_configuration=lambda FunctionName: {"State": "Active", "PackageType": "Image"},
                           get_function_url_config=lambda FunctionName: {"AuthType": "AWS_IAM"})
        result = verify.Result()
        verify.check_function(result, client, items, {"ApiBaseUrl": "https://x.lambda-url.r.on.aws/"}, True,
                              lambda url: {"model_name": "m", "bundle_sha256": "a" * 64, "synthetic_model": False})
        self.assertEqual(set(statuses(result).values()), {"PASS"})

    def test_an_unreadable_resource_is_reported_not_raised(self):
        s3 = Namespace(get_bucket_versioning=lambda Bucket: (_ for _ in ()).throw(RuntimeError("AccessDenied")))
        result = verify.Result()
        verify.check_buckets(result, s3, [{"type": "AWS::S3::Bucket", "physical": "x-artifacts"}] * 2)
        self.assertTrue(result.failed())
        self.assertIn("AccessDenied", result.table())

    def test_the_latest_pipeline_run_is_reported(self):
        items = [{"type": "AWS::StepFunctions::StateMachine", "physical": "arn:sm"}]
        for latest, expected in (("SUCCEEDED", "PASS"), ("RUNNING", "WARN"), ("FAILED", "FAIL"), (None, "WARN")):
            runs = [{"name": "run-1", "status": latest}] if latest else []
            sfn = Namespace(describe_state_machine=lambda stateMachineArn: {"status": "ACTIVE"},
                            list_executions=lambda stateMachineArn, maxResults, runs=runs: {"executions": runs})
            result = verify.Result()
            verify.check_pipeline(result, sfn, items)
            self.assertEqual(statuses(result)["Pipeline run"], expected, latest)

    def test_a_zero_training_quota_is_flagged_before_the_pipeline_fails(self):
        quotas = [{"QuotaName": "ml.m5.xlarge for training job usage", "Value": 0.0},
                  {"QuotaName": "ml.m5.xlarge for spot training job usage", "Value": 2.0}]
        client = Namespace(get_paginator=lambda _: Namespace(paginate=lambda ServiceCode: [{"Quotas": quotas}]))
        result = verify.Result()
        verify.check_quotas(result, client)
        self.assertEqual(sorted(statuses(result).values()), ["PASS", "WARN"])
        self.assertIn("request an increase", result.table())

    def test_a_low_fargate_quota_is_flagged_before_a_training_task_fails(self):
        quotas = [{"QuotaName": "Fargate On-Demand vCPU resource count", "Value": 2.0},
                  {"QuotaName": "Fargate Spot vCPU resource count", "Value": 6.0}]
        client = Namespace(get_paginator=lambda _: Namespace(paginate=lambda ServiceCode: [{"Quotas": quotas}]))
        result = verify.Result()
        verify.check_fargate_quota(result, client, needed_vcpu=4)
        self.assertEqual(statuses(result), {"Fargate quota (on demand)": "WARN", "Fargate quota (spot)": "PASS"})
        self.assertIn("Raise it in Service Quotas", result.table())

    def test_the_ecs_training_cluster_is_checked_only_when_it_exists(self):
        result = verify.Result()
        verify.check_ecs(result, Namespace(), [])
        self.assertEqual(statuses(result), {})
        items = [{"type": "AWS::ECS::Cluster", "physical": "sepsis-dev-training-cluster"}]
        for providers, expected in ((["FARGATE", "FARGATE_SPOT"], "PASS"), (["FARGATE"], "FAIL")):
            ecs = Namespace(describe_clusters=lambda clusters, providers=providers: {
                "clusters": [{"status": "ACTIVE", "capacityProviders": providers}]})
            result = verify.Result()
            verify.check_ecs(result, ecs, items)
            self.assertEqual(list(statuses(result).values()), [expected])

    def test_the_hosted_tools_are_running_or_stopped_never_silently_missing(self):
        items = [{"type": "AWS::ECS::Service", "logical": "Service", "physical": "arn:aws:ecs:r:1:service/c/sepsis-dev-mlflow"},
                 {"type": "AWS::RDS::DBInstance", "logical": "Database", "physical": "db-1"}]
        for running, desired, db, expected in ((1, 1, "available", ["PASS", "PASS"]), (0, 0, "stopped", ["WARN", "WARN"]),
                                               (0, 1, "available", ["FAIL", "PASS"])):
            ecs = Namespace(describe_services=lambda cluster, services, running=running, desired=desired: {"services": [
                {"runningCount": running, "desiredCount": desired, "events": [{"message": "unable to place a task"}]}]})
            rds = Namespace(describe_db_instances=lambda DBInstanceIdentifier, db=db: {"DBInstances": [{"DBInstanceStatus": db}]})
            result = verify.Result()
            verify.check_tools(result, ecs, rds, items)
            self.assertEqual(list(statuses(result).values()), expected)

    def test_a_missing_service_linked_role_is_reported_with_the_command_that_fixes_it(self):
        class NoSuchEntity(Exception):
            response = {"Error": {"Code": "NoSuchEntity"}}

        def get_role(RoleName):
            if RoleName == "AWSServiceRoleForECS":
                raise NoSuchEntity("The role cannot be found")
            return {"Role": {}}

        result = verify.Result()
        verify.check_service_linked_roles(result, Namespace(get_role=get_role), {"TrainingBackend": "ecs", "MlflowBackend": "ecs"})
        self.assertEqual(statuses(result), {"Service-linked role AWSServiceRoleForECS": "FAIL", "Service-linked role AWSServiceRoleForRDS": "PASS"})
        self.assertIn("create-service-linked-role --aws-service-name ecs.amazonaws.com", result.table())
        nothing = verify.Result()
        verify.check_service_linked_roles(nothing, Namespace(), {"TrainingBackend": "sagemaker", "MlflowBackend": "sagemaker"})
        self.assertEqual(statuses(nothing), {})

    def test_the_scoring_image_is_not_required_in_lambda_mode_but_is_with_fargate_serving(self):
        items = [{"type": "AWS::ECR::Repository", "logical": "EcrRepository", "physical": "scoring-repo"},
                 {"type": "AWS::ECR::Repository", "logical": "LambdaRepository", "physical": "lambda-repo"},
                 {"type": "AWS::ECR::Repository", "logical": "TrainingRepository", "physical": "training-repo"}]
        ecr = Namespace(describe_images=lambda repositoryName: {"imageDetails": (
            [] if repositoryName == "scoring-repo" else [{"imageTags": ["lambda-abc123def456"]}])})
        lambda_mode = verify.Result()
        verify.check_repositories(lambda_mode, ecr, items, "abc123def456", "lambda")
        self.assertEqual(statuses(lambda_mode)["ECR scoring-repo"], "PASS")
        fargate = verify.Result()
        verify.check_repositories(fargate, ecr, items, "abc123def456", "fargate")
        self.assertEqual(statuses(fargate)["ECR scoring-repo"], "FAIL")
        self.assertEqual(statuses(fargate)["ECR lambda-repo"], "PASS")

    def test_missing_outputs_fail(self):
        result = verify.Result()
        verify.check_outputs(result, {"ArtifactBucketName": "b"}, False)
        self.assertTrue(result.failed())

    def test_script_and_workflows(self):
        self.assertIn("aws_verify.py", (ROOT / "scripts" / "aws_deploy.sh").read_text(encoding="utf-8"))
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "verify.yml").read_text(encoding="utf-8"))
        self.assertEqual(workflow["permissions"]["id-token"], "write")
        self.assertIn("ap-southeast-2", workflow[True]["workflow_dispatch"]["inputs"]["region"]["options"])
        deploy = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8"))
        names = [step.get("name") for step in deploy["jobs"]["deploy"]["steps"]]
        self.assertIn("Verify every deployed component", names)
        # Only the failure diagnosis and the summary of next steps may come after the verification.
        self.assertEqual(names[names.index("Verify every deployed component") + 1:], ["Diagnose the failure", "What to do next"])


if __name__ == "__main__":
    unittest.main()
