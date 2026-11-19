#!/usr/bin/env python3
"""Print why a CloudFormation stack failed, including its nested stacks.

    python scripts/aws_stack_failures.py --stack sepsis-dev --region ap-southeast-2

`aws cloudformation deploy` only says "Failed to create/update the stack". The reason is in the events
of the nested stack that failed, which are easy to miss (and the nested stacks are deleted by the rollback,
though their events stay readable by stack ID). This lists the first real failures, oldest first, and
skips the "Resource creation cancelled" lines that are only a side effect. Needs boto3.
"""

import argparse
import os
import sys


def events_of(cloudformation, stack: str) -> list[dict]:
    found, token = [], None
    while True:
        page = cloudformation.describe_stack_events(StackName=stack, **({"NextToken": token} if token else {}))
        found += page["StackEvents"]
        token = page.get("NextToken")
        if not token:
            return found


def validation_failures(cloudformation, stack: str) -> list[dict]:
    """Template checks CloudFormation ran before creating anything (for example "a bucket with this name exists").

    The stack events only say "Validation failed with N error(s)"; the details are in DescribeEvents.
    """
    try:
        page = cloudformation.describe_events(StackName=stack, Filters={"FailedEvents": True})
    except Exception:  # noqa: BLE001  an older SDK or CLI without this API
        return []
    found = []
    for event in page.get("OperationEvents", []):
        reason = event.get("ValidationStatusReason") or event.get("ResourceStatusReason")
        if reason and (event.get("ValidationName") or event.get("ValidationStatus")):
            where = event.get("ValidationPath") or event.get("LogicalResourceId") or ""
            found.append({"stack": stack, "logical": f"{event.get('ValidationName', 'validation')} {where}".strip(),
                          "type": event.get("ResourceType", "validation"), "reason": reason, "time": event.get("Timestamp")})
    return found


def failures(cloudformation, stack: str, seen: set | None = None) -> list[dict]:
    seen = seen if seen is not None else set()
    if stack in seen:
        return []
    seen.add(stack)
    try:
        events = events_of(cloudformation, stack)
    except Exception as error:  # noqa: BLE001  a nested stack we may not be allowed to read
        return [{"stack": stack, "logical": "(events unreadable)", "type": "", "reason": str(error), "time": None}]
    found = validation_failures(cloudformation, stack)
    for event in events:
        status, reason = event.get("ResourceStatus", ""), event.get("ResourceStatusReason", "")
        if status.endswith("FAILED") and reason and "cancelled" not in reason.lower():
            found.append({"stack": event.get("StackName", stack), "logical": event["LogicalResourceId"],
                          "type": event.get("ResourceType", ""), "reason": reason, "time": event.get("Timestamp")})
        physical = event.get("PhysicalResourceId", "")
        if event.get("ResourceType") == "AWS::CloudFormation::Stack" and ":stack/" in physical and physical != stack:
            found += failures(cloudformation, physical, seen)
    return found


def report(items: list[dict]) -> str:
    if not items:
        return "No failed resource with a reason was found in the stack events."
    # Nested-stack summaries ("Embedded stack ... was not successfully created") repeat the cause; show the leaf causes first.
    leaves = [i for i in items if i["type"] != "AWS::CloudFormation::Stack"] or items
    leaves.sort(key=lambda i: (i["time"] is None, i["time"]))
    lines = ["Why the stack failed (first failure first):"]
    for item in leaves:
        lines.append(f"  {item['stack'].split('/')[1] if '/' in item['stack'] else item['stack']} · {item['logical']} "
                     f"({item['type']}): {item['reason']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", default=None)
    args = parser.parse_args(argv)
    import boto3

    text = report(failures(boto3.Session(region_name=args.region).client("cloudformation"), args.stack))
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("### Why the stack failed\n\n```\n" + text + "\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
