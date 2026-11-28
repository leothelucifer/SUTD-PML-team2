#!/usr/bin/env python3
"""List and validate the source files used by CI's type and coverage gates."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "mypy": (
        "src/sepsis_warning/serving/api.py",
        "src/sepsis_warning/serving/service.py",
        "src/sepsis_warning/serving/client.py",
        "src/sepsis_warning/serving/alert_store.py",
        "src/sepsis_warning/monitoring/telemetry.py",
        "src/sepsis_warning/lifecycle/gates.py",
        "src/sepsis_warning/lifecycle/manifest.py",
        "src/sepsis_warning/adapters/aws/lambda_app.py",
        "src/sepsis_warning/lifecycle/model_source.py",
        "src/sepsis_warning/monitoring/live_drift.py",
        "src/sepsis_warning/adapters/aws/drift_handler.py",
    ),
    "coverage": (
        "src/sepsis_warning/features/core.py",
        "src/sepsis_warning/models/policy.py",
        "src/sepsis_warning/serving/service.py",
        "src/sepsis_warning/serving/api.py",
        "src/sepsis_warning/serving/client.py",
        "src/sepsis_warning/serving/alert_store.py",
        "src/sepsis_warning/lifecycle/gates.py",
        "src/sepsis_warning/lifecycle/manifest.py",
        "src/sepsis_warning/evaluation/frozen.py",
        "src/sepsis_warning/monitoring/live_drift.py",
        "src/sepsis_warning/monitoring/telemetry.py",
        "src/sepsis_warning/lifecycle/model_source.py",
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=TARGETS)
    args = parser.parse_args(argv)
    targets = TARGETS[args.gate]
    missing = [path for path in targets if not (ROOT / path).is_file()]
    if missing:
        parser.error(f"{args.gate} targets do not exist: {', '.join(missing)}")
    print("\n".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
