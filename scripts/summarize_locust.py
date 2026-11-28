#!/usr/bin/env python3
"""Turn Locust aggregate statistics into a patient-data-free JSON and Markdown report."""

import argparse
import csv
import json
from pathlib import Path


def _number(row: dict[str, str], key: str, integer: bool = False):
    value = row.get(key, "")
    if not value:
        raise ValueError(f"Locust stats row is missing {key!r}")
    return int(float(value)) if integer else float(value)


def aggregate_stats(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    aggregate = next(
        (row for row in rows if row.get("Name") == "Aggregated" or row.get("Type") == "Aggregated"),
        None,
    )
    if aggregate is None:
        aggregate = next((row for row in rows if row.get("Name") == "POST /predict"), None)
    if aggregate is None:
        raise ValueError("Locust stats contain neither an Aggregated row nor POST /predict")
    requests = _number(aggregate, "Request Count", integer=True)
    failures = _number(aggregate, "Failure Count", integer=True)
    return {
        "requests": requests,
        "failures": failures,
        "successful_requests": requests - failures,
        "failure_rate_percent": round(100 * failures / requests, 3) if requests else None,
        "requests_per_second": _number(aggregate, "Requests/s"),
        "average_ms": _number(aggregate, "Average Response Time"),
        "median_ms": _number(aggregate, "Median Response Time"),
        "p95_ms": _number(aggregate, "95%"),
        "p99_ms": _number(aggregate, "99%"),
        "min_ms": _number(aggregate, "Min Response Time"),
        "max_ms": _number(aggregate, "Max Response Time"),
        "average_response_bytes": _number(aggregate, "Average Content Size"),
    }


def build_report(stats_path: Path, manifest_path: Path, health_path: Path, metadata: dict) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    health = json.loads(health_path.read_text(encoding="utf-8"))
    if health.get("synthetic_model") is not False:
        raise ValueError("Refusing to report a performance run unless /health confirms a non-synthetic model")
    return {
        **metadata,
        "endpoint": "POST /predict",
        "data": {
            "cohort": manifest["cohort"],
            "source": manifest["source"],
            "records": manifest["records"],
            "hours": manifest["hours"],
        },
        "model": {
            "name": health.get("model_name"),
            "version": health.get("model_version"),
            "bundle_sha256": health.get("bundle_sha256"),
        },
        "metrics": aggregate_stats(stats_path),
    }


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    model = report["model"]
    return "\n".join([
        "# Locust API performance results",
        "",
        f"- Run: `{report['run_id']}` ({report['environment']}, `{report['region']}`)",
        f"- Endpoint: `{report['endpoint']}`",
        f"- Model: `{model['name']}` version `{model['version']}`; synthetic: no",
        f"- Hospital B records: {report['data']['records']} ({report['data']['hours']} hourly rows)",
        f"- Load: {report['users']} users, spawn rate {report['spawn_rate']}/s, requested duration {report['duration']}",
        "",
        "| Requests | Failures | Failure rate | Requests/s | Average ms | Median ms | p95 ms | p99 ms | Min ms | Max ms | Avg bytes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {metrics['requests']} | {metrics['failures']} | {metrics['failure_rate_percent']}% | "
        f"{metrics['requests_per_second']} | {metrics['average_ms']} | {metrics['median_ms']} | "
        f"{metrics['p95_ms']} | {metrics['p99_ms']} | {metrics['min_ms']} | {metrics['max_ms']} | "
        f"{metrics['average_response_bytes']} |",
        "",
        "Only aggregate request metrics and model provenance are retained. The Hospital B payloads and hourly rows are temporary and are not included.",
        "",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--spawn-rate", type=float, required=True)
    parser.add_argument("--duration", required=True)
    args = parser.parse_args(argv)

    report = build_report(
        args.stats, args.manifest, args.health,
        {
            "run_id": args.run_id,
            "environment": args.environment,
            "region": args.region,
            "users": args.users,
            "spawn_rate": args.spawn_rate,
            "duration": args.duration,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
