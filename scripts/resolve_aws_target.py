#!/usr/bin/env python3
"""Resolve the selected AWS region and OIDC role for a GitHub Actions job."""

import os
from pathlib import Path

REGIONS = {"ap-southeast-1", "ap-southeast-2"}
ENVIRONMENTS = {"dev", "staging", "prod"}


def resolve_target(environment: str, region_input: str, default_region: str,
                   account_id: str, role_arn: str) -> tuple[str, str]:
    if environment not in ENVIRONMENTS:
        raise ValueError(f"unsupported environment: {environment!r}")
    region = default_region if region_input == "environment default" else region_input
    if region not in REGIONS:
        raise ValueError(f"unsupported region: {region!r}")
    role_arn = role_arn.strip()
    account_id = account_id.strip()
    if "\n" in role_arn or "\r" in role_arn:
        raise ValueError("AWS_DEPLOY_ROLE_ARN must be a single-line ARN")
    if role_arn:
        role = role_arn
    elif account_id and account_id.isdigit() and len(account_id) == 12:
        role = f"arn:aws:iam::{account_id}:role/github-deploy-{environment}-{region}"
    elif account_id:
        raise ValueError("AWS_ACCOUNT_ID must contain exactly 12 digits")
    else:
        role = ""
    if not role:
        raise ValueError(
            f"environment {environment!r} has neither AWS_ACCOUNT_ID nor AWS_DEPLOY_ROLE_ARN; "
            "run scripts/bootstrap_github_oidc.sh or configure the GitHub environment"
        )
    return region, role


def main() -> int:
    try:
        region, role = resolve_target(
            os.environ.get("ENVIRONMENT", ""),
            os.environ.get("REGION_INPUT", ""),
            os.environ.get("DEFAULT_REGION", ""),
            os.environ.get("ACCOUNT_ID", ""),
            os.environ.get("ROLE_ARN_VAR", ""),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        raise SystemExit("GITHUB_OUTPUT is not set")
    with Path(output).open("a", encoding="utf-8") as handle:
        handle.write(f"region={region}\nrole={role}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
