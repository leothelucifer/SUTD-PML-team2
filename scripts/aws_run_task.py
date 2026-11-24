#!/usr/bin/env python3
"""Run one step of the project as a one-off ECS Fargate task and wait for it (used by the promote workflow).

    python scripts/aws_run_task.py --stack sepsis-dev --region ap-southeast-2 --role approver \\
        --image-tag training-<commit> --env HP_mode=promote --env HP_model_version=3 --env HP_reviewed_by=alice

The step runs the training image's ``sepsis_warning.ecs_job`` entry point inside the project's VPC, so it can
reach the private MLflow server. ``--role approver`` runs it as the approver task role, which is a different
principal from the training task role: the one that registered a model version cannot also promote it.

Everything it needs (cluster, subnets, security group, roles, image repository, log group) is read from the
stack outputs. It registers a task definition for the image tag, runs it, waits for it to stop, prints the
task's log and returns the container's exit code.
"""

import argparse
import os
import sys
import time


def stack_outputs(cloudformation, stack: str) -> dict:
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}


def needed(outputs: dict, *names: str) -> dict:
    missing = [n for n in names if not outputs.get(n)]
    if missing:
        raise ValueError(f"the stack has no output {', '.join(missing)}; deploy it with an ECS backend first")
    return {n: outputs[n] for n in names}


def parse_env(pairs: list[str]) -> list[dict]:
    items = []
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise ValueError(f"--env needs NAME=VALUE, got {pair!r}")
        items.append({"name": name, "value": value})
    return items


def register(ecs, family: str, image: str, execution_role: str, task_role: str, log_group: str, region: str,
             cpu: str = "1024", memory: str = "4096") -> str:
    response = ecs.register_task_definition(
        family=family, cpu=cpu, memory=memory, networkMode="awsvpc", requiresCompatibilities=["FARGATE"],
        executionRoleArn=execution_role, taskRoleArn=task_role,
        runtimePlatform={"cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX"},
        containerDefinitions=[{
            "name": "step", "image": image, "essential": True,
            "entryPoint": ["python", "-m", "sepsis_warning.ecs_job"],
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": log_group, "awslogs-region": region, "awslogs-stream-prefix": family}},
        }])
    return response["taskDefinition"]["taskDefinitionArn"]


def log_location(ecs, definition: str, container: str) -> tuple[str, str]:
    """(log group, stream prefix) of a container in an existing task definition."""
    containers = ecs.describe_task_definition(taskDefinition=definition)["taskDefinition"]["containerDefinitions"]
    options = next(c for c in containers if c["name"] == container)["logConfiguration"]["options"]
    return options["awslogs-group"], options["awslogs-stream-prefix"]


def run(session, stack: str, *, role: str, image_tag: str, env: list[str], family_suffix: str, region: str,
        sleep=time.sleep, timeout: int = 1800, out=print, existing_family: str = "", container: str = "step") -> int:
    cloudformation, ecs, logs = session.client("cloudformation"), session.client("ecs"), session.client("logs")
    outputs = stack_outputs(cloudformation, stack)
    if existing_family:
        # A task definition the stack already owns (the Evidently report): run its latest revision as it is.
        o = needed(outputs, "EcsClusterArn", "EcsSubnetIds", "EcsTaskSecurityGroupId")
        definition, family = existing_family, existing_family
        log_group, log_prefix = log_location(ecs, existing_family, container)
    else:
        o = needed(outputs, "EcsClusterArn", "EcsSubnetIds", "EcsTaskSecurityGroupId", "EcsExecutionRoleArn",
                   "EcsTaskRoleArn", "EcsApproverTaskRoleArn", "EcsLogGroupName", "TrainingRepositoryUri")
        task_role = o["EcsApproverTaskRoleArn"] if role == "approver" else o["EcsTaskRoleArn"]
        family = f"{stack}-{family_suffix}"
        definition = register(ecs, family, f"{o['TrainingRepositoryUri']}:{image_tag}", o["EcsExecutionRoleArn"],
                              task_role, o["EcsLogGroupName"], region)
        log_group, log_prefix = o["EcsLogGroupName"], f"{family}/step"
    started = ecs.run_task(
        cluster=o["EcsClusterArn"], taskDefinition=definition, launchType="FARGATE", count=1,
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": o["EcsSubnetIds"].split(","), "securityGroups": [o["EcsTaskSecurityGroupId"]],
            "assignPublicIp": "ENABLED"}},
        overrides={"containerOverrides": [{"name": container, "environment": parse_env(env)}]})
    if started.get("failures"):
        raise RuntimeError(f"the task could not be started: {started['failures']}")
    task_arn = started["tasks"][0]["taskArn"]
    out(f"Started {task_arn.rsplit('/', 1)[-1]} as {role} on {o['EcsClusterArn'].rsplit('/', 1)[-1]}")
    deadline = time.monotonic() + timeout
    while True:
        task = ecs.describe_tasks(cluster=o["EcsClusterArn"], tasks=[task_arn])["tasks"][0]
        if task["lastStatus"] == "STOPPED":
            break
        if time.monotonic() > deadline:
            ecs.stop_task(cluster=o["EcsClusterArn"], task=task_arn, reason="timed out")
            raise TimeoutError(f"the task did not finish within {timeout} seconds and was stopped")
        sleep(10)
    finished = task["containers"][0]
    task_id = task_arn.rsplit("/", 1)[-1]
    stream = f"{log_prefix}/{container}/{task_id}" if existing_family else f"{log_prefix}/{task_id}"
    show_log(logs, log_group, stream, out)
    code = finished.get("exitCode")
    if code is None:
        out(f"The task stopped without an exit code: {task.get('stoppedReason', finished.get('reason', 'unknown'))}")
        return 1
    return int(code)


def show_log(logs, group: str, stream: str, out=print, limit: int = 60) -> None:
    try:
        events = logs.get_log_events(logGroupName=group, logStreamName=stream, limit=limit, startFromHead=False)["events"]
    except Exception as error:  # noqa: BLE001 - a missing log must not hide the exit code
        out(f"(log not readable: {type(error).__name__})")
        return
    for event in events:
        out(event["message"].rstrip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--role", choices=["task", "approver"], default="task")
    parser.add_argument("--image-tag", default="", help="training-<git commit> (not needed with --existing-family)")
    parser.add_argument("--existing-family", default="", help="run the latest revision of a task definition the stack owns")
    parser.add_argument("--container", default="step", help="container name to override (report for the Evidently report)")
    parser.add_argument("--family", default="step", help="task definition family suffix")
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE for the task (HP_*, CH_*, JOB_OUTPUT)")
    args = parser.parse_args(argv)
    import boto3

    try:
        code = run(boto3.Session(region_name=args.region), args.stack, role=args.role, image_tag=args.image_tag,
                   env=args.env, family_suffix=args.family, region=args.region,
                   existing_family=args.existing_family, container=args.container)
    except (ValueError, RuntimeError, TimeoutError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Task {args.family}: {'succeeded' if code == 0 else f'failed (exit {code})'}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
