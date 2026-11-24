#!/usr/bin/env python3
"""Approve or reject the model that the pipeline is waiting on (used by the approve workflow).

    python scripts/aws_approval.py --stack sepsis-dev --region ap-southeast-2 --decision approve --by alice

The pipeline stops at WaitForApproval and holds a task token. This script finds the waiting execution (the
named one, or the only one), reads the token from the execution history, and answers it:

* approve  ->  send-task-success with {"approved": true, "by": <who>}; the frozen evaluation and the publish follow.
* reject   ->  send-task-failure; the pipeline ends and a failure notice is sent.

The name of the person who answered is stored in the execution's own output, so the decision is on the audit
trail next to the step that waited for it. Whether the model was promoted by a reviewer is checked by the next
step itself, before the one-time test evaluation, so approving a version that was never promoted is refused there.
"""

import argparse
import json
import sys


def waiting_executions(stepfunctions, state_machine: str) -> list[dict]:
    found = []
    for item in stepfunctions.list_executions(stateMachineArn=state_machine, statusFilter="RUNNING").get("executions", []):
        events = stepfunctions.get_execution_history(executionArn=item["executionArn"], reverseOrder=True, maxResults=50)["events"]
        token = None
        for event in events:                        # newest first: a task still waiting is the latest scheduled one
            if event["type"] == "TaskSucceeded" or event["type"] == "TaskFailed":
                break
            details = event.get("taskScheduledEventDetails")
            if details and details.get("resource") == "publish.waitForTaskToken":
                token = json.loads(details["parameters"]).get("MessageAttributes", {}).get("token", {}).get("StringValue")
                break
        if token:
            found.append({"name": item["name"], "arn": item["executionArn"], "token": token})
    return found


def choose(executions: list[dict], name: str | None) -> dict:
    if name:
        matching = [e for e in executions if e["name"] == name]
        if not matching:
            raise ValueError(f"no execution named {name!r} is waiting for approval; waiting: {[e['name'] for e in executions] or 'none'}")
        return matching[0]
    if not executions:
        raise ValueError("no pipeline execution is waiting for approval")
    if len(executions) > 1:
        raise ValueError(f"several executions are waiting ({[e['name'] for e in executions]}); name one")
    return executions[0]


def decide(stepfunctions, state_machine: str, decision: str, by: str, name: str | None = None, reason: str = "") -> str:
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be approve or reject")
    if not by.strip():
        raise ValueError("the name of the person deciding is required")
    target = choose(waiting_executions(stepfunctions, state_machine), name)
    if decision == "approve":
        stepfunctions.send_task_success(taskToken=target["token"], output=json.dumps({"approved": True, "by": by.strip()}))
    else:
        stepfunctions.send_task_failure(taskToken=target["token"], error="Rejected",
                                        cause=f"rejected by {by.strip()}" + (f": {reason[:200]}" if reason else ""))
    return target["name"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--decision", choices=["approve", "reject"], required=True)
    parser.add_argument("--by", required=True, help="who is deciding (the workflow passes the GitHub user)")
    parser.add_argument("--execution", default="", help="execution name (the run id in the approval email); default: the only waiting one")
    parser.add_argument("--reason", default="")
    args = parser.parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    try:
        described = session.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]
        outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
        if not outputs.get("PipelineArn"):
            raise ValueError("the stack has no pipeline")
        name = decide(session.client("stepfunctions"), outputs["PipelineArn"], args.decision, args.by, args.execution or None, args.reason)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"{args.decision}d execution {name} as {args.by}" if args.decision == "approve" else f"rejected execution {name} as {args.by}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
