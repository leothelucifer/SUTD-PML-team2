#!/usr/bin/env python3
"""Watch the hosted tools (MLflow, Evidently UI) while a deploy runs, and gate the deploy on them being healthy.

    python scripts/aws_watch_services.py watch --stack sepsis-dev --region ap-southeast-2   # during the stack update
    python scripts/aws_watch_services.py ready --stack sepsis-dev --region ap-southeast-2   # after it, as a gate

Why it exists: CloudFormation waits for an ECS service to reach a steady state and says nothing while it waits. When the
container crashes at start-up (a missing driver, a wrong setting) the deploy log is silent for many minutes and the reason is
only in the task's stopped reason and its log. This prints that reason as soon as a task stops.

* ``watch`` runs next to ``aws cloudformation deploy``. It prints each service's state when it changes, and for every task that
  stops it prints the stopped reason, the container's exit code and the last lines of its log. If a service has no running task
  and several tasks have stopped, it is crash-looping: the watcher cancels the stack update (so the deploy fails in about a
  minute with the cause above it, instead of waiting for the circuit breaker) and exits 2.
* ``ready`` waits until every service that is meant to run (desired count above 0) has all its tasks running and healthy.
  A service switched off by the power workflow (desired 0) is skipped. Exit 1 after the timeout, with the same diagnosis.

Service, cluster and log names follow the templates (``<stack>-training-cluster``, ``<stack>-mlflow``, ``<stack>-evidently-ui``);
the log group and stream come from each task's own definition, so nothing here repeats them.
"""

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta

CRASH_LOOP_STOPPED_TASKS = 3   # distinct tasks that stopped with nothing running before the update is cancelled
LOG_LINES = 15


def names(stack: str) -> tuple[str, list[str]]:
    return f"{stack}-training-cluster", [f"{stack}-mlflow", f"{stack}-evidently-ui"]


def describe(ecs, cluster: str, services: list[str]) -> dict[str, dict]:
    """Services that exist, by name. A service that is not created yet (or was removed) is simply absent."""
    try:
        found = ecs.describe_services(cluster=cluster, services=services)["services"]
    except Exception as error:  # noqa: BLE001 - no cluster yet is an answer, not a failure
        if "ClusterNotFound" in type(error).__name__ or "ClusterNotFound" in str(error):
            return {}
        raise
    return {s["serviceName"]: s for s in found if s.get("status") != "INACTIVE"}


def line(service: dict) -> str:
    events = service.get("events") or [{}]
    return (f"{service['serviceName']}: running {service.get('runningCount', 0)}/{service.get('desiredCount', 0)}, "
            f"pending {service.get('pendingCount', 0)}; {events[0].get('message', '')[:140]}")


def stopped_tasks(ecs, cluster: str, service: str, since: datetime | None = None) -> list[dict]:
    """Tasks of the service that stopped (ECS keeps them for about an hour). ``since`` leaves out the ones from before
    this deploy, so an earlier crash does not make a fixed service look like it is crash-looping."""
    arns = ecs.list_tasks(cluster=cluster, serviceName=service, desiredStatus="STOPPED").get("taskArns", [])
    tasks = ecs.describe_tasks(cluster=cluster, tasks=arns[:20])["tasks"] if arns else []
    return [t for t in tasks if since is None or (t.get("stoppedAt") and t["stoppedAt"] >= since)]


def log_tail(ecs, logs, cluster: str, task: dict, lines: int = LOG_LINES) -> list[str]:
    """Last lines of the task's first container, using the log group and stream prefix its own task definition names."""
    try:
        definition = ecs.describe_task_definition(taskDefinition=task["taskDefinitionArn"])["taskDefinition"]
        container = definition["containerDefinitions"][0]
        options = container["logConfiguration"]["options"]
        stream = f"{options['awslogs-stream-prefix']}/{container['name']}/{task['taskArn'].rsplit('/', 1)[-1]}"
        events = logs.get_log_events(logGroupName=options["awslogs-group"], logStreamName=stream, limit=lines,
                                     startFromHead=False)["events"]
        return [e["message"].rstrip() for e in events]
    except Exception as error:  # noqa: BLE001 - a log that is not there yet must not hide the stopped reason
        return [f"(log not readable: {type(error).__name__})"]


def diagnosis(ecs, logs, cluster: str, task: dict) -> list[str]:
    containers = task.get("containers") or [{}]
    first = containers[0]
    code = first.get("exitCode")
    out = [f"  task {task['taskArn'].rsplit('/', 1)[-1]} stopped: {task.get('stoppedReason', 'no reason given')}"
           + (f" (exit code {code})" if code is not None else "")
           + (f" - {first['reason']}" if first.get("reason") else "")]
    out += [f"    | {message}" for message in log_tail(ecs, logs, cluster, task)]
    return out


def crash_looping(service: dict, stopped: list[dict]) -> bool:
    """Nothing running and several different tasks have already stopped for this service."""
    return service.get("runningCount", 0) == 0 and len(stopped) >= CRASH_LOOP_STOPPED_TASKS


def watch(ecs, logs, cloudformation, stack: str, *, interval: float = 30, sleep=time.sleep, out=print, once: bool = False,
          cancel: bool = True, clock=lambda: datetime.now(UTC), since_hours: float = 0) -> int:
    cluster, services = names(stack)
    since = clock() - timedelta(hours=since_hours)
    last: dict[str, str] = {}
    reported: set[str] = set()
    while True:
        for name, service in describe(ecs, cluster, services).items():
            text = line(service)
            if last.get(name) != text:
                last[name] = text
                out(f"[watch] {text}")
            stopped = stopped_tasks(ecs, cluster, name, since) if service.get("runningCount", 0) == 0 else []
            for task in stopped:
                if task["taskArn"] not in reported:
                    reported.add(task["taskArn"])
                    for message in diagnosis(ecs, logs, cluster, task):
                        out(f"[watch] {message}")
            if crash_looping(service, stopped):
                out(f"[watch] {name} is crash-looping: {len(stopped)} tasks stopped and none is running.")
                if cancel:
                    out(f"[watch] cancelling the update of {stack} so the deploy fails now, not after the circuit breaker")
                    try:
                        cloudformation.cancel_update_stack(StackName=stack)
                    except Exception as error:  # noqa: BLE001 - e.g. the stack is being created, which cannot be cancelled
                        out(f"[watch] could not cancel the update ({type(error).__name__}); the circuit breaker will end it")
                return 2
        if once:
            return 0
        sleep(interval)


def ready(ecs, logs, stack: str, *, timeout: float = 900, interval: float = 15, sleep=time.sleep, clock=time.monotonic,
          out=print) -> int:
    cluster, services = names(stack)
    deadline = clock() + timeout
    while True:
        found = describe(ecs, cluster, services)
        wanted = {n: s for n, s in found.items() if s.get("desiredCount", 0) > 0}
        waiting = []
        for name, service in wanted.items():
            tasks = ecs.list_tasks(cluster=cluster, serviceName=name, desiredStatus="RUNNING").get("taskArns", [])
            described = ecs.describe_tasks(cluster=cluster, tasks=tasks)["tasks"] if tasks else []
            healthy = [t for t in described if t.get("healthStatus", "UNKNOWN") in ("HEALTHY", "UNKNOWN") and t["lastStatus"] == "RUNNING"]
            bad = [t for t in described if t.get("healthStatus") == "UNHEALTHY"]
            if service["runningCount"] < service["desiredCount"] or len(healthy) < service["desiredCount"] or bad:
                waiting.append(name)
        for name, service in found.items():
            if service.get("desiredCount", 0) == 0:
                out(f"{name}: switched off (desired 0), not checked")
        if not waiting:
            for name, service in wanted.items():
                out(f"{name}: running {service['runningCount']}/{service['desiredCount']} and healthy")
            if not wanted:
                out("no hosted service is meant to be running")
            return 0
        if clock() > deadline:
            for name in waiting:
                out(f"NOT READY {line(wanted[name])}")
                for task in stopped_tasks(ecs, cluster, name)[:2]:
                    for message in diagnosis(ecs, logs, cluster, task):
                        out(message)
            return 1
        out("waiting for " + ", ".join(f"{n} ({wanted[n]['runningCount']}/{wanted[n]['desiredCount']})" for n in waiting))
        sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["watch", "ready"])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--once", action="store_true", help="watch only: print the state and the stopped tasks once, then exit")
    parser.add_argument("--no-cancel", action="store_true", help="watch only: report a crash loop but leave the update running")
    args = parser.parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    ecs, logs = session.client("ecs"), session.client("logs")
    if args.mode == "watch":
        return watch(ecs, logs, session.client("cloudformation"), args.stack, interval=args.interval, cancel=not args.no_cancel,
                     once=args.once, since_hours=1 if args.once else 0)
    return ready(ecs, logs, args.stack, timeout=args.timeout, interval=min(args.interval, 15))


if __name__ == "__main__":
    sys.exit(main())
