#!/usr/bin/env python3
"""Replay real stays through the deployed scoring endpoint and keep the ward page in S3 (the predict demo).

    python scripts/aws_demo.py --stack sepsis-dev --region ap-southeast-2 --split validation

What it does, in order:

1. calls the deployed function URL (SigV4-signed, the way a hospital system would) for ``/health`` and reports which
   model is serving;
2. for a held-out split (``internal_test`` or ``external_test``, i.e. hospital B) it reads the published model's
   ``registry.json`` and refuses unless that model has a recorded frozen evaluation, so test data is only looked at
   after the model was frozen, and passes ``--after-freeze`` to the replay itself;
3. downloads the prepared PhysioNet data from the artifact bucket, sends the stays hour by hour to ``POST /score``
   and writes the ward replay page (alerts against the recorded onset);
4. uploads the page to ``s3://<bucket>/reports/demo/<time>/`` and prints where it is.

The page holds patient-level trajectories from PhysioNet, so it stays in the private encrypted bucket. It is
never written into the repository and never uploaded as a workflow artifact.
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HELD_OUT = ("internal_test", "external_test")
SPLITS = ("train", "calibration", "validation") + HELD_OUT


def require_frozen(split: str, registry: dict | None) -> None:
    """A held-out split may only be replayed for a model with a recorded frozen evaluation."""
    if split not in HELD_OUT:
        return
    if not registry or not registry.get("frozen_evaluated_at"):
        raise ValueError(
            f"refusing to replay {split}: the serving model has no recorded frozen evaluation. Approve it in the pipeline "
            "(the one evaluation of the test splits runs after the approval) before showing test data")


def ward_argv(*, api_url: str, data: Path, splits: Path, split: str, beds: int, seed: int, output: Path) -> list[str]:
    argv = ["ward", "--api-url", api_url, "--sign", "--data", str(data), "--splits", str(splits), "--split", split,
            "--beds", str(beds), "--seed", str(seed), "--output", str(output)]
    if split in HELD_OUT:
        argv.append("--after-freeze")
    return argv


def report_key(now: datetime.datetime, split: str) -> str:
    return f"reports/demo/{now.strftime('%Y%m%dT%H%M%SZ')}/ward-{split}.html"


def read_json(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as error:  # noqa: BLE001 - a missing file is an answer, not a crash
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def run(session, stack: str, region: str, *, split: str, beds: int, seed: int, data_folder: str, out=print) -> str:
    from aws_verify import signed_health

    cloudformation, s3 = session.client("cloudformation"), session.client("s3")
    described = cloudformation.describe_stacks(StackName=stack)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
    api, bucket = outputs.get("ApiBaseUrl"), outputs.get("ArtifactBucketName")
    if not api or not bucket:
        raise ValueError("the stack has no scoring endpoint yet: run the deploy workflow with steps 'full deploy' first")
    health = signed_health(session, region)(api.rstrip("/") + "/health")
    out(f"Serving {health.get('model_name')} version {health.get('model_version')} "
        f"(bundle {str(health.get('bundle_sha256'))[:12]}, synthetic={health.get('synthetic_model')})")
    if health.get("synthetic_model"):
        raise ValueError("the endpoint serves a synthetic model; this demo replays real PhysioNet stays")
    version = health.get("model_version")
    registry = read_json(s3, bucket, f"models/v{version}/registry.json") if version else None
    require_frozen(split, registry)
    with tempfile.TemporaryDirectory(prefix="sepsis-demo-") as temp:
        temp = Path(temp)
        for name in ("combined.parquet", "splits.csv"):
            s3.download_file(bucket, f"data/{data_folder}/{name}", str(temp / name))
        page = temp / "ward.html"
        argv = ward_argv(api_url=api.rstrip("/"), data=temp / "combined.parquet", splits=temp / "splits.csv",
                         split=split, beds=beds, seed=seed, output=page)
        from sepsis_warning import cli

        old_argv = sys.argv
        sys.argv = ["sepsis-warning", *argv]
        try:
            cli.main()
        except SystemExit as stop:      # the CLI reports a bad request with parser.error()
            if stop.code not in (0, None):
                raise RuntimeError(f"the ward replay was refused (exit {stop.code}); see the message above") from stop
        finally:
            sys.argv = old_argv
        key = report_key(datetime.datetime.now(datetime.UTC), split)
        s3.upload_file(str(page), bucket, key, ExtraArgs={"ContentType": "text/html"})
    location = f"s3://{bucket}/{key}"
    out(f"Ward replay saved to {location} (private; open it with: aws s3 cp {location} . )")
    return location


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--beds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-folder", default="physionet", help="data/<folder>/ in the artifact bucket")
    args = parser.parse_args(argv)
    import boto3

    try:
        location = run(boto3.Session(region_name=args.region), args.stack, args.region, split=args.split,
                       beds=args.beds, seed=args.seed, data_folder=args.data_folder)
    except (ValueError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Predict demo\n\nWard replay ({args.split}): `{location}`\n\nPrivate: it holds PhysioNet stays.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
