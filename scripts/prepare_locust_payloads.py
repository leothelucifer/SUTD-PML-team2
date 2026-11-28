#!/usr/bin/env python3
"""Prepare private, anonymized Hospital B payloads for the Locust API test."""

import argparse
import json
import os
import random
from pathlib import Path

from sepsis_warning.data.records import DYNAMIC, STATIC, read_patient


def prepare_payloads(data_root: Path, limit: int, seed: int) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    files = sorted((data_root / "training_setB").glob("*.psv"))
    if not files:
        raise ValueError(f"No Hospital B .psv records in {data_root / 'training_setB'}")

    selected = random.Random(seed).sample(files, min(limit, len(files)))
    columns = DYNAMIC + STATIC
    payloads = []
    for index, path in enumerate(selected):
        frame = read_patient(path)
        payloads.append({
            "patient_id": f"hospital-b-{index:04d}",
            "hours": json.loads(frame[columns].to_json(orient="records")),
        })
    return payloads


def write_private_jsonl(path: Path, payloads: list[dict]) -> None:
    repository = Path(__file__).resolve().parents[1]
    path = path.resolve()
    if path.is_relative_to(repository):
        raise ValueError("payload output must be outside the repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        for payload in payloads:
            output.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    os.chmod(path, 0o600)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Downloaded dataset root; only training_setB is read")
    parser.add_argument("--output", type=Path, required=True, help="Private JSONL output path outside the repository")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of B stays to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    payloads = prepare_payloads(args.data_root, args.limit, args.seed)
    write_private_jsonl(args.output, payloads)
    manifest = {
        "cohort": "hospital B",
        "source": "PhysioNet Challenge 2019",
        "records": len(payloads),
        "hours": sum(len(payload["hours"]) for payload in payloads),
        "seed": args.seed,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
