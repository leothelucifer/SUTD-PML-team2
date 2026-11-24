#!/usr/bin/env python3
"""Find the model the pipeline already published to S3, for the deploy workflow's "full deploy".

    python scripts/aws_published_model.py --stack sepsis-dev --region ap-southeast-2 [--version 3]

With MLflow on ECS the registry is private (inside the VPC), so the deploy job cannot call it. It does not need to: the
pipeline's last step has already approved, evaluated and published the model to a write-once folder,
``s3://<bucket>/models/v<N>/``, and recorded who approved it in ``registry.json`` next to the bundle. This script reads that
folder, prints the two values the serving stack needs in the same form ``python -m sepsis_warning publish`` does
(``ModelS3Uri=...`` and ``ModelSha256=...``), and refuses a real-data model that was not promoted and evaluated.

No SageMaker and no MLflow call is made.
"""

import argparse
import json
import re
import sys


def versions(s3, bucket: str, prefix: str = "models") -> list[int]:
    found, token = set(), None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{prefix}/v", "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for item in page.get("CommonPrefixes", []):
            match = re.search(r"/v(\d+)/$", item["Prefix"])
            if match:
                found.add(int(match.group(1)))
        if not page.get("IsTruncated"):
            return sorted(found)
        token = page.get("NextContinuationToken")


def read_json(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as error:  # noqa: BLE001 - a missing file is an answer
        if getattr(error, "response", {}).get("Error", {}).get("Code", "") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def find(s3, bucket: str, version: int | None = None, prefix: str = "models") -> dict:
    available = versions(s3, bucket, prefix)
    if not available:
        raise ValueError(f"no published model under s3://{bucket}/{prefix}/: run the pipeline and approve it first "
                         "(the last step publishes the approved model)")
    chosen = version if version is not None else available[-1]
    if chosen not in available:
        raise ValueError(f"version {chosen} is not published; published versions: {available}")
    key = f"{prefix}/v{chosen}/model.joblib"
    metadata = s3.head_object(Bucket=bucket, Key=key).get("Metadata", {})
    sha = metadata.get("sha256")
    if not sha or len(sha) != 64:
        raise ValueError(f"s3://{bucket}/{key} has no SHA-256 in its metadata; it was not published by the pipeline")
    registry = read_json(s3, bucket, f"{prefix}/v{chosen}/registry.json") or {}
    return {"version": chosen, "uri": f"s3://{bucket}/{key}", "sha256": sha, "registry": registry}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--version", type=int, default=None, help="published version (default: the newest)")
    parser.add_argument("--allow-synthetic", action="store_true", help="accept a model that was not promoted (rehearsals)")
    args = parser.parse_args(argv)
    import boto3

    session = boto3.Session(region_name=args.region)
    try:
        described = session.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]
        outputs = {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}
        found = find(session.client("s3"), outputs["ArtifactBucketName"], args.version)
        registry = found["registry"]
        if not args.allow_synthetic and not (registry.get("approved_by") and registry.get("frozen_evaluated_at")):
            raise ValueError(f"version {found['version']} has no recorded approval and frozen evaluation (registry.json: "
                             f"{sorted(registry) or 'missing'}); approve it in the pipeline, or pass --allow-synthetic for a rehearsal")
    except (ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Published version {found['version']}, approved by {registry.get('approved_by') or 'nobody (rehearsal)'}, "
          f"frozen evaluation {registry.get('frozen_evaluated_at') or 'none'}")
    print(f"Deploy it with: ModelS3Uri={found['uri']} ModelSha256={found['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
