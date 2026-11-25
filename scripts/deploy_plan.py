#!/usr/bin/env python3
"""Plan a deploy from what changed, and remember what each stage last did.

    python scripts/deploy_plan.py plan   --stack sepsis-dev --region ap-southeast-2 --head <sha> [--mode auto] [--retrain auto]
    python scripts/deploy_plan.py record --stack sepsis-dev --region ap-southeast-2 --head <sha> --stages stack,serve

The deploy workflow is one manual button. This decides what it does:

* ``state.json`` (``s3://<artifact bucket>/deployments/<env>/state.json``) holds, per stage (stack, retrain, serve), the commit
  it last ran for, and the model version last served.
* For each stage it takes the files changed since that stage's commit and asks the rules (ci/change_rules.json, the same ones
  CI uses) whether they need it. No state means the stage never ran here: it is needed.
* Then the facts of the account adjust that: serving needs a published (approved) model, or a newer one than the one served;
  a retrain is skipped while a pipeline run is in progress, and when a model already exists in a fresh environment; nothing but
  the stack can run before the stack exists.
* ``mode`` and ``retrain`` override the automatic choice (the manual options). ``--dry-run`` prints the plan and runs nothing.

Stages the workflow then runs, in order: validate, images, stack, ready, retrain, serve (smoke test), verify. ``record`` is run
last, only when everything passed, and writes the commit of the stages that ran. Needs boto3 (not for the planning logic).
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_changes  # noqa: E402

STAGES = ("stack", "retrain", "serve")
MODES = ("auto", "stack and images only", "full deploy")
RETRAIN = ("auto", "yes", "no")


def short(commit: str | None) -> str:
    return (commit or "")[:7]


def needed_by_changes(state: dict | None, head: str, rules: dict, files_between=ci_changes.changed_files) -> dict:
    """{stage: (needed, reason)} from the files changed since each stage last ran."""
    result = {}
    for stage in STAGES:
        base = ((state or {}).get("stages") or {}).get(stage)
        if not base:
            result[stage] = (True, "never ran in this environment")
            continue
        files = files_between(base, head)
        found = ci_changes.plan(files, rules)
        if not found["known"]:
            result[stage] = (True, f"cannot compare with {short(base)}; assuming changed")
        elif stage in found["deploy"]:
            result[stage] = (True, f"changed since {short(base)}: {', '.join(found['categories'])}")
        else:
            result[stage] = (False, f"nothing relevant changed since {short(base)}")
    return result


def decide(*, state: dict | None, head: str, rules: dict, mode: str, retrain: str, stack_exists: bool,
           model_versions: list[int], pipeline_running: bool, files_between=ci_changes.changed_files) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if retrain not in RETRAIN:
        raise ValueError(f"retrain must be one of {RETRAIN}")
    changes = needed_by_changes(state, head, rules, files_between)
    reasons = {stage: reason for stage, (_, reason) in changes.items()}
    notes: list[str] = []
    latest = max(model_versions) if model_versions else None
    served = (state or {}).get("model_version")

    run = {stage: changes[stage][0] for stage in STAGES}
    if mode == "stack and images only":
        run = {"stack": True, "retrain": False, "serve": False}
        reasons["stack"] = "chosen: stack and images only"
    elif mode == "full deploy":
        run["stack"] = True
        run["serve"] = True
        reasons["stack"] = reasons["serve"] = "chosen: full deploy"
        run["retrain"] = changes["retrain"][0] and latest is None
    if not stack_exists:
        run["stack"] = True
        reasons["stack"] = "the stack does not exist yet"

    # Serving needs an approved, published model, and is also due when a newer one exists than the one served.
    if run["serve"] or mode == "auto":
        if latest is None:
            if run["serve"]:
                notes.append("Serving skipped: no approved model is published yet. Train, promote and approve one first.")
            run["serve"] = False
        elif mode == "auto" and (served is None or latest > served) and stack_exists:
            run["serve"] = True
            reasons["serve"] = f"model v{latest} is published and v{served} is the one served" if served else f"model v{latest} is published and none is served yet"
    if not stack_exists:
        run["serve"] = False

    # Retraining: the explicit choice wins, then the changes. A fresh environment that already has a model does not retrain.
    if retrain == "yes":
        run["retrain"] = True
        reasons["retrain"] = "chosen: retrain"
    elif retrain == "no":
        run["retrain"] = False
        reasons["retrain"] = "chosen: no retrain"
    elif mode == "auto" and run["retrain"] and not (state or {}).get("stages", {}).get("retrain") and latest is not None:
        run["retrain"] = False
        reasons["retrain"] = f"a model (v{latest}) is already published; nothing to train from scratch"
    if run["retrain"] and mode == "stack and images only":
        run["retrain"] = False
    if run["retrain"] and not stack_exists:
        run["retrain"] = False
        notes.append("Training needs the stack: deploy again after this run to start it.")
    if run["retrain"] and pipeline_running:
        run["retrain"] = False
        notes.append("Retrain skipped: a pipeline run is in progress (it may be waiting for your approval).")

    flags = {
        "stack": run["stack"], "ready": run["stack"], "retrain": run["retrain"], "serve": run["serve"],
        "images": run["stack"] or run["retrain"] or run["serve"],
    }
    flags["validate"] = flags["stack"] or flags["serve"]
    flags["verify"] = flags["images"]
    flags["any"] = flags["images"]
    return {"flags": flags, "reasons": reasons, "notes": notes, "latest_model": latest, "served_model": served}


def markdown(result: dict, dry_run: bool) -> str:
    flags, reasons = result["flags"], result["reasons"]
    rows = [("stack", "Images, stack update, wait for MLflow and Evidently to be healthy"),
            ("retrain", "Copy real data if missing, start the pipeline (it stops at your approval)"),
            ("serve", "Publish the approved model, serve it on Lambda, smoke test")]
    lines = ["### Deploy plan" + (" (dry run: nothing is executed)" if dry_run else ""), "", "| Stage | Runs | What | Why |", "|---|---|---|---|"]
    for stage, what in rows:
        lines.append(f"| {stage} | {'yes' if flags[stage] else 'no'} | {what} | {reasons.get(stage, '')} |")
    if not flags["any"]:
        lines += ["", "Nothing to do: everything is already up to date."]
    for note in result["notes"]:
        lines += ["", f"- {note}"]
    return "\n".join(lines) + "\n"


def write_outputs(result: dict, dry_run: bool, output: str | None, summary: str | None) -> None:
    flags = {k: (v and not dry_run) for k, v in result["flags"].items()}
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            for name, value in flags.items():
                handle.write(f"{name}={'true' if value else 'false'}\n")
            ran = [s for s in STAGES if flags[s]]
            handle.write(f"stages={','.join(ran)}\n")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown(result, dry_run))


# ---- the account's facts (injected in tests)

def read_state(s3, bucket: str, environment: str) -> dict | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=f"deployments/{environment}/state.json")["Body"].read()
    except Exception as error:  # noqa: BLE001 - no bucket and no file both mean "no state"
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "NoSuchBucket", "404", "NotFound"):
            return None
        raise
    return json.loads(body)


def write_state(s3, bucket: str, environment: str, state: dict) -> None:
    s3.put_object(Bucket=bucket, Key=f"deployments/{environment}/state.json", Body=json.dumps(state, indent=2).encode(),
                  ContentType="application/json")


def record(state: dict | None, head: str, stages: list[str], model_version: int | None, by: str, run: str,
           now: datetime | None = None) -> dict:
    state = dict(state or {})
    done = dict(state.get("stages") or {})
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        done[stage] = head
    state.update({"version": 1, "stages": done, "updated_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
                  "by": by, "run": run})
    if "serve" in stages and model_version is not None:
        state["model_version"] = model_version
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["plan", "record"])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--mode", default="auto", choices=MODES)
    parser.add_argument("--retrain", default="auto", choices=RETRAIN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", default="")
    parser.add_argument("--by", default="")
    parser.add_argument("--run", default="")
    args = parser.parse_args(argv)
    import boto3
    from aws_published_model import versions

    session = boto3.Session(region_name=args.region)
    environment = args.stack.removeprefix("sepsis-")
    outputs, stack_exists = {}, True
    try:
        described = session.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]
        outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
    except Exception as error:  # noqa: BLE001
        if "does not exist" not in str(error):
            raise
        stack_exists = False
    bucket, s3 = outputs.get("ArtifactBucketName", ""), session.client("s3")
    state = read_state(s3, bucket, environment) if bucket else None
    published = versions(s3, bucket) if bucket else []

    if args.action == "record":
        stages = [s for s in args.stages.split(",") if s]
        write_state(s3, bucket, environment, record(state, args.head, stages, max(published) if published else None, args.by, args.run))
        print(f"recorded {args.head[:7]} for {', '.join(stages) or 'no stages'}")
        return 0

    running = False
    if outputs.get("PipelineArn"):
        running = bool(session.client("stepfunctions").list_executions(
            stateMachineArn=outputs["PipelineArn"], statusFilter="RUNNING", maxResults=1).get("executions"))
    result = decide(state=state, head=args.head, rules=ci_changes.load_rules(), mode=args.mode, retrain=args.retrain,
                    stack_exists=stack_exists, model_versions=published, pipeline_running=running)
    write_outputs(result, args.dry_run, os.environ.get("GITHUB_OUTPUT"), os.environ.get("GITHUB_STEP_SUMMARY"))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
