#!/usr/bin/env python3
"""Check that every component of a deployment exists and is healthy.

    python scripts/aws_verify.py --stack sepsis-dev --region ap-southeast-2
    python scripts/aws_verify.py --stack sepsis-dev --region ap-southeast-2 --expect-model --tag <git sha>

It reads the stack (and its nested stacks) and then asks each service about the real resource, so a stack
that reports CREATE_COMPLETE but has an empty image repository, an unencrypted bucket or an unconfirmed alert
email still shows up. Read-only: nothing is created, changed or deleted.

Each line is PASS, WARN (works, but needs a human, e.g. confirming an email) or FAIL. The exit code is 1 when
anything FAILs. Without --expect-model the scoring function is not required (a "stack and images only" deploy
has none yet); with it, the function, its IAM-protected URL and a signed /health call are checked as well.
Needs boto3 (aws extra). In GitHub Actions the table is also written to the run summary.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

OK_STATUSES = ("CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE", "IMPORT_COMPLETE")


class Result:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, component: str, detail: str = "") -> None:
        self.rows.append((status, component, detail))

    def failed(self) -> bool:
        return any(status == "FAIL" for status, _, _ in self.rows)

    def table(self) -> str:
        width = max((len(c) for _, c, _ in self.rows), default=10)
        lines = [f"{status:<5} {component:<{width}}  {detail}".rstrip() for status, component, detail in self.rows]
        counts = {s: sum(1 for r in self.rows if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
        lines.append(f"\n{counts['PASS']} passed, {counts['WARN']} need attention, {counts['FAIL']} failed")
        return "\n".join(lines)

    def markdown(self) -> str:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        lines = ["### Deployment check", "", "| | Component | Detail |", "|---|---|---|"]
        lines += [f"| {icon[s]} | {c} | {d} |" for s, c, d in self.rows]
        return "\n".join(lines) + "\n"


def resources(cloudformation, stack: str, seen: set | None = None) -> list[dict]:
    """Every resource of the stack and its nested stacks: {type, logical, physical, status, stack}."""
    seen = seen if seen is not None else set()
    if stack in seen:
        return []
    seen.add(stack)
    found, token = [], None
    while True:
        page = cloudformation.list_stack_resources(StackName=stack, **({"NextToken": token} if token else {}))
        for item in page["StackResourceSummaries"]:
            entry = {"type": item["ResourceType"], "logical": item["LogicalResourceId"],
                     "physical": item.get("PhysicalResourceId", ""), "status": item["ResourceStatus"], "stack": stack}
            found.append(entry)
            if entry["type"] == "AWS::CloudFormation::Stack" and entry["physical"]:
                found += resources(cloudformation, entry["physical"], seen)
        token = page.get("NextToken")
        if not token:
            return found


def of_type(items: list[dict], kind: str) -> list[dict]:
    return [i for i in items if i["type"] == kind]


def guarded(result: Result, component: str, call) -> None:
    try:
        call()
    except Exception as error:  # noqa: BLE001  a missing resource or a permission problem is a finding, not a crash
        result.add("FAIL", component, f"{type(error).__name__}: {str(error)[:160]}")


def check_stack(result: Result, cloudformation, stack: str, items: list[dict]) -> None:
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    status = described["StackStatus"]
    result.add("PASS" if status in OK_STATUSES else "FAIL", f"stack {stack}", status)
    for nested in of_type(items, "AWS::CloudFormation::Stack"):
        name = nested["physical"].split("/")[1] if "/" in nested["physical"] else nested["logical"]
        result.add("PASS" if nested["status"] in OK_STATUSES else "FAIL", f"  nested stack {nested['logical']}", f"{name}: {nested['status']}")
    bad = [i for i in items if i["status"] not in OK_STATUSES and i["type"] != "AWS::CloudFormation::Stack"]
    for item in bad:
        result.add("FAIL", f"  resource {item['logical']}", f"{item['type']}: {item['status']}")


def check_buckets(result: Result, s3, items: list[dict]) -> None:
    buckets = of_type(items, "AWS::S3::Bucket")
    if len(buckets) < 2:
        result.add("FAIL", "S3 buckets", f"expected the artifact and log buckets, found {len(buckets)}")
    for bucket in buckets:
        name = bucket["physical"]

        def check(name=name):
            versioning = s3.get_bucket_versioning(Bucket=name).get("Status")
            encryption = s3.get_bucket_encryption(Bucket=name)["ServerSideEncryptionConfiguration"]["Rules"][0][
                "ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]
            block = s3.get_public_access_block(Bucket=name)["PublicAccessBlockConfiguration"]
            blocked = all(block.get(k) for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"))
            problems = []
            if not blocked:
                problems.append("public access not fully blocked")
            if name.endswith("-artifacts") and versioning != "Enabled":
                problems.append("versioning off")
            result.add("FAIL" if problems else "PASS", f"S3 {name}",
                       "; ".join(problems) if problems else f"encrypted ({encryption}), public access blocked, versioning {versioning or 'off'}")

        guarded(result, f"S3 {name}", check)


def check_key(result: Result, kms, items: list[dict]) -> None:
    keys = of_type(items, "AWS::KMS::Key")
    if not keys:
        result.add("FAIL", "KMS key", "not found in the stack")
    for key in keys:
        def check(key=key):
            meta = kms.describe_key(KeyId=key["physical"])["KeyMetadata"]
            rotation = kms.get_key_rotation_status(KeyId=key["physical"])["KeyRotationEnabled"]
            good = meta["KeyState"] == "Enabled" and rotation
            result.add("PASS" if good else "FAIL", f"KMS key {key['physical'][:8]}…", f"{meta['KeyState']}, rotation {'on' if rotation else 'off'}")

        guarded(result, "KMS key", check)


def check_repositories(result: Result, ecr, items: list[dict], tag: str | None) -> None:
    repos = of_type(items, "AWS::ECR::Repository")
    if len(repos) < 3:
        result.add("FAIL", "ECR repositories", f"expected 3, found {len(repos)}")
    for repo in repos:
        name = repo["physical"]

        def check(name=name):
            images = ecr.describe_images(repositoryName=name).get("imageDetails", [])
            tags = sorted({t for image in images for t in image.get("imageTags", [])})
            if not images:
                result.add("FAIL", f"ECR {name}", "no image pushed (run the images step)")
            elif tag and not any(t == tag or t.endswith("-" + tag) for t in tags):
                # images are tagged <kind>-<commit> (training-, lambda-, scoring-), so match the commit at the end
                result.add("FAIL", f"ECR {name}", f"{len(images)} image(s) but none for commit {tag[:12]}")
            else:
                result.add("PASS", f"ECR {name}", f"{len(images)} image(s), including this commit's")

        guarded(result, f"ECR {name}", check)


def check_table(result: Result, dynamodb, items: list[dict]) -> None:
    for table in of_type(items, "AWS::DynamoDB::Table"):
        def check(table=table):
            described = dynamodb.describe_table(TableName=table["physical"])["Table"]
            sse = (described.get("SSEDescription") or {}).get("Status")
            ok = described["TableStatus"] == "ACTIVE"
            result.add("PASS" if ok else "FAIL", f"DynamoDB {table['physical']}", f"{described['TableStatus']}, encryption {sse or 'default'}")

        guarded(result, f"DynamoDB {table['physical']}", check)


def check_mlflow(result: Result, sagemaker, items: list[dict]) -> None:
    apps = of_type(items, "AWS::SageMaker::MlflowApp")
    if not apps:
        result.add("WARN", "SageMaker MLflow App", "not in this stack (MlflowBackend=ecs, or EnableMlops is off)")
    for app in apps:
        def check(app=app):
            status = sagemaker.describe_mlflow_app(Arn=app["physical"])["Status"]
            result.add("PASS" if status in ("Created", "Updated") else "FAIL", "SageMaker MLflow App", status)

        guarded(result, "SageMaker MLflow App", check)


def check_ecs(result: Result, ecs, items: list[dict]) -> None:
    """With TrainingBackend=ecs the pipeline steps run on this cluster (Fargate and Fargate Spot)."""
    clusters = of_type(items, "AWS::ECS::Cluster")
    if not clusters:
        return    # TrainingBackend=sagemaker (or Fargate serving, whose cluster check lives with the service)
    for cluster in clusters:
        def check(cluster=cluster):
            described = ecs.describe_clusters(clusters=[cluster["physical"]])["clusters"]
            if not described:
                result.add("FAIL", f"ECS cluster {cluster['physical']}", "not found")
                return
            status = described[0]["status"]
            providers = set(described[0].get("capacityProviders", []))
            # The Fargate serving cluster has no capacity providers of its own; only the training one needs them.
            missing = ({"FARGATE", "FARGATE_SPOT"} - providers) if "training" in cluster["physical"] else set()
            ok = status == "ACTIVE" and not missing
            result.add("PASS" if ok else "FAIL", f"ECS cluster {cluster['physical']}",
                       f"{status}, capacity providers {sorted(providers)}" + (f", missing {sorted(missing)}" if missing else ""))

        guarded(result, f"ECS cluster {cluster['physical']}", check)


def check_tools(result: Result, ecs, rds, items: list[dict]) -> None:
    """The hosted tools (MLflow server, Evidently UI) and the database. Stopped is a normal, cheap state: a WARN."""
    for item in of_type(items, "AWS::ECS::Service"):
        def check(item=item):
            cluster, _, name = item["physical"].split("service/", 1)[-1].partition("/")
            service = ecs.describe_services(cluster=cluster, services=[name])["services"][0]
            running, desired = service["runningCount"], service["desiredCount"]
            if running >= 1 and running >= desired:
                result.add("PASS", f"ECS service {name}", f"running {running}/{desired}")
            elif desired == 0:
                result.add("WARN", f"ECS service {name}", "stopped (start it with the power workflow when you need it)")
            else:
                result.add("FAIL", f"ECS service {name}", f"running {running} of {desired}: {service.get('events', [{}])[0].get('message', 'see the service events')[:120]}")

        guarded(result, f"ECS service {item['logical']}", check)
    for item in of_type(items, "AWS::RDS::DBInstance"):
        def check(item=item):
            status = rds.describe_db_instances(DBInstanceIdentifier=item["physical"])["DBInstances"][0]["DBInstanceStatus"]
            result.add("PASS" if status == "available" else "WARN", f"RDS {item['physical']}",
                       status if status == "available" else f"{status} (start it with the power workflow when you need it)")

        guarded(result, f"RDS {item['logical']}", check)


def check_pipeline(result: Result, stepfunctions, items: list[dict]) -> None:
    machines = of_type(items, "AWS::StepFunctions::StateMachine")
    if not machines:
        result.add("WARN", "Step Functions approval pipeline", "not in this stack (EnablePipeline is off)")
    for machine in machines:
        def check(machine=machine):
            status = stepfunctions.describe_state_machine(stateMachineArn=machine["physical"])["status"]
            result.add("PASS" if status == "ACTIVE" else "FAIL", "Step Functions approval pipeline", status)
            runs = stepfunctions.list_executions(stateMachineArn=machine["physical"], maxResults=1).get("executions", [])
            if not runs:
                result.add("WARN", "Pipeline run", "never run (start the pipeline workflow)")
                return
            last = runs[0]["status"]
            detail = f"latest run {runs[0].get('name', '')[:36]}: {last}"
            if last == "SUCCEEDED":
                result.add("PASS", "Pipeline run", detail)
            elif last == "RUNNING":
                result.add("WARN", "Pipeline run", detail + " (running, or waiting for the approval)")
            else:
                result.add("FAIL", "Pipeline run", detail)

        guarded(result, "Step Functions approval pipeline", check)


def check_alerts(result: Result, sns, cloudwatch, items: list[dict]) -> None:
    topics = of_type(items, "AWS::SNS::Topic")
    if not topics:
        result.add("WARN", "SNS alert topic", "not found in the stack")
    for topic in topics:
        def check(topic=topic):
            subs = sns.list_subscriptions_by_topic(TopicArn=topic["physical"]).get("Subscriptions", [])
            pending = [s for s in subs if s["SubscriptionArn"] == "PendingConfirmation"]
            if not subs:
                result.add("WARN", f"SNS {topic['logical']}", "no email subscribed (set ALERT_EMAIL to get alarm and approval emails)")
            elif pending:
                result.add("WARN", f"SNS {topic['logical']}", f"{len(pending)} subscription(s) waiting for the email confirmation link")
            else:
                result.add("PASS", f"SNS {topic['logical']}", f"{len(subs)} confirmed subscription(s)")

        guarded(result, f"SNS {topic['logical']}", check)
    names = [a["physical"] for a in of_type(items, "AWS::CloudWatch::Alarm")]
    if names:
        def alarms():
            states = {}
            for start in range(0, len(names), 100):
                for alarm in cloudwatch.describe_alarms(AlarmNames=names[start:start + 100])["MetricAlarms"]:
                    states[alarm["AlarmName"]] = alarm["StateValue"]
            firing = [n for n, v in states.items() if v == "ALARM"]
            missing = [n for n in names if n not in states]
            if firing or missing:
                result.add("FAIL", "CloudWatch alarms", f"firing: {firing or 'none'}; missing: {missing or 'none'}")
            else:
                result.add("PASS", "CloudWatch alarms", f"{len(states)} alarm(s), none firing")

        guarded(result, "CloudWatch alarms", alarms)


def check_quotas(result: Result, service_quotas, instance_type: str = "ml.m5.xlarge") -> None:
    """SageMaker training instances start at 0 in a new account; the pipeline cannot run until they are raised."""
    wanted = {f"{instance_type} for training job usage": "on demand (the data download)",
              f"{instance_type} for spot training job usage": "spot (tune, train, evaluate)"}

    def check():
        found = {}
        for page in service_quotas.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for quota in page["Quotas"]:
                if quota["QuotaName"] in wanted:
                    found[quota["QuotaName"]] = quota["Value"]
        for name, purpose in wanted.items():
            value = found.get(name)
            if value is None:
                result.add("WARN", f"SageMaker quota {name}", "not found; check Service Quotas")
            elif value < 1:
                result.add("WARN", f"SageMaker quota {name}", f"{value:g}: request an increase to 1 or more in Service Quotas ({purpose})")
            else:
                result.add("PASS", f"SageMaker quota {name}", f"{value:g} ({purpose})")

    guarded(result, "SageMaker quotas", check)


def check_fargate_quota(result: Result, service_quotas, needed_vcpu: float = 5.0) -> None:
    """ECS training runs on Fargate. The default quota is 6 vCPU per Region for On-Demand and for Spot (the ECS quotas page);
    one training task (4) plus the MLflow server (0.5) and the Evidently UI (0.5) need 5 of them at the same time."""
    def check():
        found = {}
        for page in service_quotas.get_paginator("list_service_quotas").paginate(ServiceCode="fargate"):
            for quota in page["Quotas"]:
                name = quota["QuotaName"]
                if "vCPU" in name and "On-Demand" in name:
                    found["on demand"] = (name, quota["Value"])
                elif "vCPU" in name and "Spot" in name:
                    found["spot"] = (name, quota["Value"])
        if not found:
            result.add("WARN", "Fargate vCPU quota", "not found; check Service Quotas (service code fargate)")
        for kind, (name, value) in sorted(found.items()):
            if value < needed_vcpu:
                result.add("WARN", f"Fargate quota ({kind})",
                           f"{value:g} vCPU; a training task plus the tools needs {needed_vcpu:g}. Raise it in Service Quotas ({name}) "
                           "or lower EcsTrainingCpu")
            else:
                result.add("PASS", f"Fargate quota ({kind})", f"{value:g} vCPU (a training task plus the tools needs {needed_vcpu:g})")

    guarded(result, "Fargate quotas", check)


def check_budget(result: Result, budgets, account: str, items: list[dict]) -> None:
    found = of_type(items, "AWS::Budgets::Budget")
    if not found:
        result.add("WARN", "AWS Budget", "none (set BUDGET_EMAIL and MonthlyBudgetUsd for a cost alert)")
    for budget in found:
        def check(budget=budget):
            described = budgets.describe_budget(AccountId=account, BudgetName=budget["physical"])["Budget"]
            limit = described["BudgetLimit"]
            result.add("PASS", "AWS Budget", f"{limit['Amount']} {limit['Unit']} {described['TimeUnit'].lower()}")

        guarded(result, "AWS Budget", check)


def check_iam(result: Result, iam, outputs: dict) -> None:
    for key in ("TrainerPolicyArn", "ApproverPolicyArn", "DeployerPolicyArn"):
        arn = outputs.get(key)
        if not arn:
            continue
        guarded(result, f"IAM {key}", lambda arn=arn, key=key: (iam.get_policy(PolicyArn=arn), result.add("PASS", f"IAM {key}", arn.split("/")[-1])))
    role = outputs.get("SageMakerExecutionRoleArn")
    if role:
        guarded(result, "IAM SageMaker role", lambda: (iam.get_role(RoleName=role.split("/")[-1]), result.add("PASS", "IAM SageMaker role", role.split("/")[-1])))


def check_outputs(result: Result, outputs: dict, expect_model: bool) -> None:
    required = ["ArtifactBucketName", "ArtifactKeyArn", "TrainingRepositoryUri", "LambdaRepositoryUri", "StateTableName"]
    missing = [k for k in required if not outputs.get(k)]
    result.add("FAIL" if missing else "PASS", "stack outputs", f"missing {missing}" if missing else f"{len(outputs)} outputs, including the bucket, key, repositories and table")
    if expect_model and not outputs.get("ApiBaseUrl"):
        result.add("FAIL", "scoring API URL", "no ApiBaseUrl output: run the full deploy (publish and serve)")


def check_function(result: Result, lambda_client, items: list[dict], outputs: dict, expect_model: bool, signed_call=None) -> None:
    functions = of_type(items, "AWS::Lambda::Function")
    scoring = [f for f in functions if "scoring" in f["physical"] or "scoring" in f["logical"].lower()]
    if not scoring:
        result.add("FAIL" if expect_model else "WARN", "Lambda scoring function",
                   "not deployed yet (expected after a full deploy)" if not expect_model else "not found in the stack")
        return
    for function in scoring:
        def check(function=function):
            config = lambda_client.get_function_configuration(FunctionName=function["physical"])
            url = lambda_client.get_function_url_config(FunctionName=function["physical"])
            ok = config.get("State") == "Active" and url.get("AuthType") == "AWS_IAM"
            result.add("PASS" if ok else "FAIL", f"Lambda {function['physical']}",
                       f"{config.get('State')}, {config.get('PackageType')}, URL auth {url.get('AuthType')}")

        guarded(result, f"Lambda {function['physical']}", check)
    base = outputs.get("ApiBaseUrl")
    if base and signed_call:
        def call():
            body = signed_call(base.rstrip("/") + "/health")
            synthetic = body.get("synthetic_model")
            result.add("PASS", "API /health (signed call)", f"model {body.get('model_name')}, bundle {str(body.get('bundle_sha256'))[:12]}, synthetic={synthetic}")

        guarded(result, "API /health (signed call)", call)


def signed_health(session, region: str):
    """A SigV4-signed GET, the same way a hospital integration would call the function URL."""
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    def call(url: str) -> dict:
        credentials = session.get_credentials()
        request = AWSRequest(method="GET", url=url, headers={})
        SigV4Auth(credentials, "lambda", region).add_auth(request)
        prepared = urllib.request.Request(url, headers=dict(request.headers))
        try:
            with urllib.request.urlopen(prepared, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}") from error

    return call


def verify(session, stack: str, region: str, *, expect_model: bool = False, tag: str | None = None, call_api: bool = True) -> Result:
    cloudformation = session.client("cloudformation")
    result = Result()
    try:
        items = resources(cloudformation, stack)
        outputs = {o["OutputKey"]: o["OutputValue"] for o in cloudformation.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])}
    except Exception as error:  # noqa: BLE001
        result.add("FAIL", f"stack {stack}", f"cannot be read in {region}: {str(error)[:160]}")
        return result
    guarded(result, f"stack {stack}", lambda: check_stack(result, cloudformation, stack, items))
    check_outputs(result, outputs, expect_model)
    check_buckets(result, session.client("s3"), items)
    check_key(result, session.client("kms"), items)
    check_repositories(result, session.client("ecr"), items, tag)
    check_table(result, session.client("dynamodb"), items)
    check_mlflow(result, session.client("sagemaker"), items)
    check_ecs(result, session.client("ecs"), items)
    check_tools(result, session.client("ecs"), session.client("rds"), items)
    check_pipeline(result, session.client("stepfunctions"), items)
    check_alerts(result, session.client("sns"), session.client("cloudwatch"), items)
    account = session.client("sts").get_caller_identity()["Account"]
    check_budget(result, session.client("budgets", region_name="us-east-1"), account, items)
    check_quotas(result, session.client("service-quotas"))
    check_fargate_quota(result, session.client("service-quotas"))
    check_iam(result, session.client("iam"), outputs)
    check_function(result, session.client("lambda"), items, outputs, expect_model, signed_health(session, region) if call_api else None)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--expect-model", action="store_true", help="Also require the scoring function and a healthy /health call")
    parser.add_argument("--tag", default=None, help="Image tag that must be present in every repository (the git commit)")
    parser.add_argument("--no-api-call", action="store_true", help="Do not call /health")
    args = parser.parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    result = verify(session, args.stack, args.region, expect_model=args.expect_model, tag=args.tag, call_api=not args.no_api_call)
    print(result.table())
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(result.markdown())
    return 1 if result.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
