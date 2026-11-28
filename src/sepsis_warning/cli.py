"""Run the sketch end to end, or score an individual patient history."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .data import generate_synthetic, load_patients, read_patient
from .data.prepare import prepare_parquet, write_summary
from .data.splits import write_splits
from .evaluation.hospital_comparison import run_comparison
from .evaluation.missingness import run_missingness_analysis
from .features import build_features, select_saved_features
from .lifecycle.tracking import (
    DEFAULT_TRACKING_URI,
    check_tracking_uri,
    require_mlflow,
)
from .models.policy import apply_policy
from .models.scoring import load_bundle
from .monitoring.offline import (
    REPLAY_SPLITS,
    MonitoringConfig,
    replay_monitoring,
    setup_monitoring,
)
from .reporting.ward import (
    HELD_OUT_SPLITS,
    SELECTION_SPLITS,
    check_split,
)
from .training.workflow import (
    run_baselines,
    run_experiment,
    run_server,
    run_sirs_failures,
    run_tune,
    run_validate,
    run_ward,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")






















DATA_HELP = "combined.parquet (uses the saved split), or a folder of training_setA/B (re-splits with --seed)"
SPLITS_HELP = "Saved split for a Parquet --data; defaults to splits.csv next to it"




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate an explicitly synthetic fixture and run it")
    demo.add_argument("--data", type=Path, default=Path("data/synthetic"))
    for command in [demo, sub.add_parser("train", help="Run on downloaded training_setA/B or combined.parquet")]:
        if command is not demo:
            command.add_argument("--data", type=Path, required=True, help=DATA_HELP)
            command.add_argument("--splits", type=Path, default=None, help=SPLITS_HELP)
        command.add_argument("--output", type=Path, default=Path("artifacts/demo" if command is demo else "artifacts/real"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--window", type=int, default=6)
        command.add_argument("--budget", type=float, default=0.2)
        command.add_argument("--cooldown", type=int, default=6)
        command.add_argument(
            "--splits-summary", type=Path, default=None,
            help="splits.csv.summary.json from `split`; its seed and hashes are logged and the seed must match",
        )
        command.add_argument("--dvc-lock", type=Path, default=None, help="dvc.lock whose output hashes are logged")
        command.add_argument(
            "--mlflow-uri", default=None,
            help=f"Local MLflow store, e.g. {DEFAULT_TRACKING_URI} (needs the tracking extra)",
        )
        command.add_argument(
            "--register-model", action="store_true",
            help="Also register the selected bundle in the MLflow registry as @candidate (needs --mlflow-uri)",
        )
        command.add_argument("--require-gates", action="store_true",
                             help="With --register-model: refuse to register a model that fails the promotion gates")
        command.add_argument("--models-config", type=Path, default=None,
                             help="Candidate models and settings (default: the packaged models.toml)")
        command.add_argument("--include-numpy-models", action="store_true",
                             help="Also fit the hand-written NumPy models as candidates (slower)")
        command.add_argument("--skip-data-checks", action="store_true",
                             help="Do not run the schema checks on a Parquet table before training")
        command.add_argument("--importance-repeats", type=int, default=3,
                             help="Shuffles per feature group for the selected model's permutation importance "
                                  "on validation (0 = skip it)")
    tune = sub.add_parser(
        "tune", help="Hyperparameter search with patient-grouped cross-validation on the training split only")
    tune.add_argument("--data", type=Path, required=True, help=DATA_HELP)
    tune.add_argument("--splits", type=Path, default=None, help=SPLITS_HELP)
    tune.add_argument("--output", type=Path, default=Path("artifacts/tuning"))
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--window", type=int, default=6)
    tune.add_argument("--folds", type=int, default=3, help="Patient-grouped cross-validation folds")
    tune.add_argument("--trials", type=int, default=12,
                      help="Sampled configurations per model, besides the reference settings")
    tune.add_argument("--max-stays", type=int, default=None,
                      help="Search on a seeded random subset of training stays (faster; recorded in the result)")
    tune.add_argument("--models", nargs="+", default=None, help="Only tune these models (default: all with a space)")
    tune.add_argument("--models-config", type=Path, default=None, help="Reference settings (default: models.toml)")
    tune.add_argument("--space-config", type=Path, default=None, help="Search space (default: tuning_space.toml)")
    tune.add_argument("--mlflow-uri", default=None,
                      help=f"Log every trial to MLflow, e.g. {DEFAULT_TRACKING_URI} (tracking extra)")
    report = sub.add_parser(
        "model-report",
        help="Metrics table, PR curves and other figures (each with its recreate command) from a finished train run")
    report.add_argument("--run", type=Path, required=True, help="Folder written by `train` (validation files)")
    report.add_argument("--output", type=Path, default=Path("reports/results"), help="New, empty folder")
    report.add_argument("--tuning", type=Path, default=None, help="Folder written by `tune` (adds the search figure)")
    report.add_argument("--evaluation", type=Path, default=None,
                        help="Folder written by `evaluate` (adds test-split table and PR curves; reads only)")
    report.add_argument("--bootstrap", type=int, default=200, help="Stay-grouped bootstrap draws (0 = no intervals)")
    report.add_argument("--seed", type=int, default=42)
    report.add_argument("--min-subgroup-septic-stays", type=int, default=10,
                        help="Subgroups with fewer septic stays are counted but not scored")
    evaluate = sub.add_parser(
        "evaluate", help="Score the test splits ONCE for a frozen model (never during model development)")
    evaluate.add_argument("--data", type=Path, required=True, help=DATA_HELP)
    evaluate.add_argument("--splits", type=Path, default=None, help=SPLITS_HELP)
    evaluate.add_argument("--seed", type=int, default=42, help="Only for folder data, which is re-split")
    evaluate_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_source.add_argument("--model", type=Path, help="model.joblib from train (manifest.json beside it)")
    evaluate_source.add_argument("--model-uri", default=None, help="Registered version, e.g. models:/sepsis-warning/3")
    evaluate.add_argument("--model-sha256", default=None)
    evaluate.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI, help="MLflow store holding --model-uri")
    evaluate.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay", help="Score a patient PSV with a saved model")
    replay.add_argument("--model", type=Path, required=True)
    replay.add_argument("--patient", type=Path, required=True)
    replay.add_argument("--output", type=Path, default=Path("artifacts/patient_replay.csv"))
    missingness = sub.add_parser(
        "missingness",
        help="Audit training/validation missingness and compare fixed baselines",
    )
    missingness.add_argument("--data", type=Path, required=True)
    missingness.add_argument(
        "--output", type=Path, default=Path("artifacts/missingness")
    )
    missingness.add_argument("--seed", type=int, default=42)
    prepare = sub.add_parser(
        "prepare", help="Validate PSV files and combine them into one Parquet file"
    )
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument(
        "--output", type=Path, default=Path("data/physionet/combined.parquet")
    )
    baseline = sub.add_parser(
        "baseline", help="Score the SIRS and always-no baselines on every evaluation split"
    )
    baseline.add_argument("--data", type=Path, required=True, help=DATA_HELP)
    baseline.add_argument("--splits", type=Path, help=SPLITS_HELP)
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--cooldown", type=int, default=6)
    baseline.add_argument("--figures", type=Path, default=Path("reports/figures"))
    validate = sub.add_parser(
        "validate", help="Check the combined Parquet against the Pandera rules (needs the checks extra)"
    )
    validate.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    validate.add_argument("--splits", type=Path, default=Path("data/physionet/splits.csv"))
    validate.add_argument("--output", type=Path, default=Path("artifacts/data_checks"))
    failures = sub.add_parser(
        "sirs-failures", help="Aggregate patterns of where SIRS fails (validation split only)"
    )
    failures.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"), help=DATA_HELP)
    failures.add_argument("--splits", type=Path, help=SPLITS_HELP)
    failures.add_argument("--output", type=Path, default=Path("artifacts/baseline/failures"))
    failures.add_argument("--seed", type=int, default=42)
    failures.add_argument("--cooldown", type=int, default=6)
    split = sub.add_parser("split", help="Save the patient split once as splits.csv")
    split.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    split.add_argument("--output", type=Path, default=Path("data/physionet/splits.csv"))
    split.add_argument("--seed", type=int, default=42)
    compare = sub.add_parser(
        "compare-hospitals",
        help="Descriptive hospital A vs B figures and summary table from the combined Parquet",
    )
    compare.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    compare.add_argument("--output", type=Path, default=Path("reports/figures"))
    compare.add_argument("--splits", type=Path, default=Path("data/physionet/splits.csv"),
                         help="Hospital A is restricted to this file's train split")
    defaults = MonitoringConfig()
    monitor_setup = sub.add_parser(
        "monitor-setup",
        help="Freeze the two-speed monitoring reference from A train/validation (run before any test replay)",
    )
    monitor_setup.add_argument("--data", type=Path, required=True)
    monitor_setup.add_argument("--model", type=Path, required=True, help="Frozen model.joblib from train/demo")
    monitor_setup.add_argument("--output", type=Path, default=Path("artifacts/monitoring/reference"))
    monitor_setup.add_argument("--seed", type=int, default=42, help="Patient split seed used by train")
    monitor_setup.add_argument("--fast-window-hours", type=int, default=defaults.fast_window_hours)
    monitor_setup.add_argument("--slow-window-hours", type=int, default=defaults.slow_window_hours)
    monitor_setup.add_argument("--min-observed-values", type=int, default=defaults.min_observed_values,
                               help="Minimum observed hourly values for a variable's PSI in one fast window")
    monitor_setup.add_argument("--min-observed-stays", type=int, default=defaults.min_observed_stays,
                               help="Minimum stays for the PSI of Age, Gender and HospAdmTime in one fast window")
    monitor_setup.add_argument("--min-slow-stays", type=int, default=defaults.min_slow_stays)
    monitor_setup.add_argument("--min-slow-septic-stays", type=int, default=defaults.min_slow_septic_stays)
    monitor_setup.add_argument("--min-subgroup-septic-stays", type=int, default=defaults.min_subgroup_septic_stays)
    monitor_setup.add_argument("--null-windows", type=int, default=defaults.null_windows)
    monitor_replay = sub.add_parser(
        "monitor-replay",
        help="Replay a held-out split through the frozen monitoring reference (aggregate outputs only)",
    )
    monitor_replay.add_argument("--data", type=Path, required=True)
    monitor_replay.add_argument("--model", type=Path, required=True)
    monitor_replay.add_argument("--reference", type=Path, required=True, help="monitoring_reference.json")
    monitor_replay.add_argument("--split", choices=REPLAY_SPLITS, default="external_test")
    monitor_replay.add_argument("--after-freeze", action="store_true",
                                help="Required for internal_test or external_test: only once the model is frozen")
    monitor_replay.add_argument("--output", type=Path, default=Path("artifacts/monitoring/replay"))
    live = sub.add_parser("monitor-live", help="Label-free drift checks on a prediction log (from serve --prediction-log)")
    live.add_argument("--predictions", type=Path, nargs="+", required=True, help="JSONL files or folders")
    live.add_argument("--reference", type=Path, required=True, help="monitoring_reference.json from monitor-setup")
    live.add_argument("--output", type=Path, default=Path("artifacts/monitoring/live"))
    serve = sub.add_parser("serve", help="Local research scoring API for a saved model (needs the serve extra)")
    serve_source = serve.add_mutually_exclusive_group(required=True)
    serve_source.add_argument("--model", type=Path, help="model.joblib produced by train/demo")
    serve_source.add_argument("--model-uri", default=None,
                              help="Registered model, e.g. models:/sepsis-warning@production (needs the tracking extra)")
    serve_source.add_argument("--model-s3", default=None,
                              help="Published bundle, s3://bucket/models/v3/model.joblib (needs --model-sha256 and the aws extra)")
    serve.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI,
                       help="MLflow store holding --model-uri: local SQLite, or the SageMaker MLflow App ARN")
    serve.add_argument("--model-sha256", default=None, help="Refuse to start unless the bundle has this SHA-256")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--alert-table", default=os.environ.get("ALERT_TABLE"),
                       help="DynamoDB alert ledger (records delivered alerts; needs the aws extra)")
    serve.add_argument("--alert-ledger", choices=["none", "memory"], default="none",
                       help="memory: keep delivered alerts in this process (lost on restart)")
    serve.add_argument("--telemetry", action="store_true",
                       help="Structured JSON request metrics and prediction lines on stdout")
    serve.add_argument("--prediction-log", type=Path, default=None,
                       help="Write prediction lines to this JSONL file (input for `monitor-live`)")
    serve.add_argument("--environment", default=os.environ.get("ENVIRONMENT_NAME", "local"))
    serve.add_argument(
        "--allow-unverified-model", action="store_true",
        help="Load a --model file that has no manifest.json and no --model-sha256 (bundles you built yourself)",
    )
    serve.add_argument(
        "--allow-non-local-host", action="store_true",
        help="Bind beyond localhost. The API has no authentication; agree this with the team first",
    )
    publish = sub.add_parser("publish", help="Copy a registered model to S3 for an AWS deployment (aws extra)")
    publish.add_argument("--model-uri", default="models:/sepsis-warning@production")
    publish.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI,
                         help="Local SQLite store or the SageMaker MLflow App ARN")
    publish.add_argument("--bucket", required=True, help="ArtifactBucketName output of the CloudFormation stack")
    publish.add_argument("--prefix", default="models")
    ward = sub.add_parser("ward", help="Write an ICU ward replay page for a saved model (research demo)")
    ward.add_argument("--model", type=Path, help="model.joblib produced by train/demo (not needed with --api-url)")
    ward.add_argument("--api-url", default=None, help="Score through a running API instead, e.g. http://127.0.0.1:8000")
    ward.add_argument("--sign", action="store_true",
                      help="Sign --api-url requests with SigV4 (the IAM-protected Lambda URL; aws extra)")
    ward.add_argument("--data", type=Path, required=True, help=DATA_HELP)
    ward.add_argument("--splits", type=Path, help=SPLITS_HELP)
    ward.add_argument("--split", choices=SELECTION_SPLITS + HELD_OUT_SPLITS, default="validation")
    ward.add_argument("--after-freeze", action="store_true",
                      help="Allow internal_test or external_test; only once the model is frozen")
    ward.add_argument("--beds", type=int, default=12)
    ward.add_argument("--septic-share", type=float, default=0.3,
                      help="Share of beds given to stays that become septic (over-represented for the demo)")
    ward.add_argument("--max-hours", type=int, default=72, help="Hours shown per stay")
    ward.add_argument("--seed", type=int, default=42)
    ward.add_argument("--output", type=Path, default=Path("artifacts/ward/ward.html"))
    promote_cmd = sub.add_parser(
        "promote", help="Point a registry alias (default production) at a reviewed model version")
    promote_cmd.add_argument("--version", required=True, help="Model version number to promote")
    promote_cmd.add_argument("--reviewed-by", required=True, help="Name of the teammate who reviewed this version")
    promote_cmd.add_argument("--alias", default="production")
    promote_cmd.add_argument("--mlflow-uri", default=DEFAULT_TRACKING_URI)
    promote_cmd.add_argument("--require-real-data", action="store_true",
                             help="Also refuse a model trained on the synthetic fixture (use for real deployments)")
    args = parser.parse_args()
    try:
        if args.command in ["demo", "train"] and args.register_model and not args.mlflow_uri:
            raise ValueError("--register-model needs --mlflow-uri, e.g. " + DEFAULT_TRACKING_URI)
        if args.command in ["demo", "train"] and args.mlflow_uri:
            # Fail before generating data or training, not after it.
            check_tracking_uri(args.mlflow_uri)
            require_mlflow()
        if args.command == "demo":
            generate_synthetic(args.data, args.seed)
            write_json(args.data / "SYNTHETIC.json", {"purpose": "Software test only", "seed": args.seed})
        if args.command in ["demo", "train"]:
            if args.window < 1 or args.budget < 0 or args.cooldown < 0:
                raise ValueError("window must be positive; budget and cooldown must be nonnegative")
            run_experiment(args)
        elif args.command == "replay":
            bundle = load_bundle(args.model)
            frame = read_patient(args.patient, require_labels=False)
            features = select_saved_features(build_features(frame, bundle["window"]), bundle["feature_columns"])
            table = pd.DataFrame({"patient_id": args.patient.stem, "hour": frame["ICULOS"]})
            with threadpool_limits(limits=2):
                table["probability"] = bundle["model"].predict_proba(features)[:, 1]
            table = apply_policy(table, bundle["threshold"], bundle["cooldown"])
            # Replay supports histories without a SepsisLabel column.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.output, index=False)
            print(args.output.resolve())
        elif args.command == "monitor-live":
            from .monitoring.live_drift import run_live_drift

            result = run_live_drift(args.predictions, args.reference, args.output)
            print(f"Live drift status: {result['status']} ({result['rows']} predictions, {result['stays']} stays)")
            for name, check in result.get("checks", {}).items():
                print(f"  {name}: {check['value'] if check['value'] is None else round(check['value'], 4)} "
                      f"[{check['status']}]")
            if result.get("detail"):
                print(f"  {result['detail']}")
            print(f"Result: {(args.output / 'live_drift.json').resolve()}", flush=True)
        elif args.command == "tune":
            if args.mlflow_uri:
                check_tracking_uri(args.mlflow_uri)
                require_mlflow()
            run_tune(args)
        elif args.command == "model-report":
            from .reporting.model_report import build_report

            manifest = build_report(args.run, args.output, tuning_dir=args.tuning, evaluation_dir=args.evaluation,
                                    bootstrap=args.bootstrap, seed=args.seed,
                                    min_subgroup_septic_stays=args.min_subgroup_septic_stays)
            print(f"Wrote {len(manifest['figures'])} figures and the metrics tables to {args.output.resolve()}")
            for item in manifest["skipped"]:
                print(f"  skipped {item['title']}: {item['reason']}")
            print(f"Open {(args.output / 'FIGURES.md').resolve()}", flush=True)
        elif args.command == "evaluate":
            from .evaluation import run_frozen_evaluation

            if args.model_uri:
                require_mlflow()
            result = run_frozen_evaluation(
                data=args.data, splits=args.splits, seed=args.seed, output=args.output, model=args.model,
                model_uri=args.model_uri, mlflow_uri=args.mlflow_uri, model_sha256=args.model_sha256,
            )
            for split, values in result["splits"].items():
                print(f"{split}: utility {values['normalized_hourly_utility']:.3f}, AUROC {values['auroc']:.3f}, "
                      f"false alerts/day {values['false_alert_events_per_nonseptic_patient_day']:.3f}", flush=True)
            again = ("a synthetic fixture, so it may be repeated" if result["data_mode"] == "synthetic_fixture"
                     else "this model cannot be evaluated again")
            print(f"Recorded as evaluated at {result['evaluated_at']}; {again}.\n"
                  f"Results: {args.output.resolve()}", flush=True)
        elif args.command == "serve":
            run_server(args)
        elif args.command == "publish":
            require_mlflow()
            from .lifecycle.model_source import publish_model

            published = publish_model(args.model_uri, args.mlflow_uri, args.bucket, args.prefix)
            print(f"Published version {published['version']} to {published['model_s3_uri']}", flush=True)
            print(f"Deploy it with: ModelS3Uri={published['model_s3_uri']} ModelSha256={published['sha256']}", flush=True)
        elif args.command == "promote":
            require_mlflow()
            from .lifecycle.registry import MODEL_NAME, promote

            change = promote(args.mlflow_uri, args.version, args.reviewed_by, alias=args.alias,
                             require_real_data=args.require_real_data)
            for gate in change["gates"]:
                print(f"  gate {gate['gate']}: passed ({gate['detail']})", flush=True)
            print(f"{MODEL_NAME}@{change['alias']} -> version {change['version']} "
                  f"(was {change['previous'] or 'unset'}; reviewed by {change['reviewed_by']}, "
                  f"approved as {change['approved_by']})", flush=True)
            if change["previous"]:
                print(f"Roll back with: python -m sepsis_warning promote --version {change['previous']} "
                      f"--alias {change['alias']} --reviewed-by <name>", flush=True)
        elif args.command == "ward":
            if not args.model and not args.api_url:
                raise ValueError("pass --model, or --api-url for a running API")
            if args.max_hours < 1:
                raise ValueError("max-hours must be positive")
            run_ward(args)
        elif args.command == "baseline":
            if args.cooldown < 0:
                raise ValueError("cooldown must be nonnegative")
            run_baselines(args)
        elif args.command == "validate":
            run_validate(args)
        elif args.command == "sirs-failures":
            if args.cooldown < 0:
                raise ValueError("cooldown must be nonnegative")
            run_sirs_failures(args)
        elif args.command == "split":
            if not args.data.exists():
                raise ValueError(f"{args.data} not found; run `prepare` first")
            summary = write_splits(args.data, args.output, args.seed)
            for name, count in summary["counts"].items():
                print(f"{name:14s} {count['stays']:6d} stays, {count['septic']:5d} septic")
            print(f"Splits: {args.output.resolve()}")
        elif args.command == "compare-hospitals":
            if not args.data.exists():
                raise ValueError(f"{args.data} not found; run `prepare` first")
            if not args.splits.exists():
                raise ValueError(f"{args.splits} not found; run `split` first")
            summary = run_comparison(args.data, args.output, args.splits)
            print(summary.to_string())
            print(f"Figures and README: {args.output.resolve()}")
        elif args.command == "missingness":
            synthetic = (args.data / "SYNTHETIC.json").exists()
            patients = load_patients(args.data)
            if not synthetic and any(
                "synthetic_" in patient.patient_id for patient in patients
            ):
                raise ValueError(
                    "Synthetic filenames found without SYNTHETIC.json marker"
                )
            summary = run_missingness_analysis(
                patients,
                args.output,
                seed=args.seed,
                synthetic=synthetic,
            )
            print(
                f"Missingness report: {(args.output / 'missingness_report.html').resolve()}"
            )
            print(
                f"Records used: train={summary['training_records']}, "
                f"validation={summary['validation_records']}"
            )
        elif args.command == "monitor-setup":
            config = MonitoringConfig(
                fast_window_hours=args.fast_window_hours, slow_window_hours=args.slow_window_hours,
                min_observed_values=args.min_observed_values, min_observed_stays=args.min_observed_stays,
                min_slow_stays=args.min_slow_stays,
                min_slow_septic_stays=args.min_slow_septic_stays,
                min_subgroup_septic_stays=args.min_subgroup_septic_stays, null_windows=args.null_windows,
            )
            reference = setup_monitoring(args.data, args.model, args.output, config, split_seed=args.seed)
            print(f"Data mode: {reference['data_mode']}; model {reference['model_name']} "
                  f"(SHA-256 {reference['model_sha256'][:12]})")
            print("Reference: A train inputs; A validation scores, alerts and performance. No test data used.")
            print(f"Monitoring reference: {(args.output / 'monitoring_reference.json').resolve()}")
        elif args.command == "monitor-replay":
            check_split(args.split, args.after_freeze)  # M01: test splits only after the model is frozen
            summary = replay_monitoring(args.data, args.model, args.reference, args.split, args.output)
            print("Research replay only; not live clinical monitoring. No automatic retraining.")
            for family, row in summary["families"].items():
                print(f"{family:16s} {row['speed']:4s} owner={row['owner']:8s} "
                      f"ok={row['ok']:3d} warn={row['warn']:3d} alert={row['alert']:3d} "
                      f"insufficient={row['insufficient_data']:3d}")
            print(f"Report: {(args.output / 'monitoring_report.md').resolve()}")
        else:
            summary = prepare_parquet(args.data, args.output)
            summary_path = args.output.with_name(
                f"{args.output.name}.summary.json"
            )
            write_summary(summary_path, summary)
            print(f"Combined Parquet: {args.output.resolve()}")
            print(
                f"Records: {summary['records']}; hourly rows: "
                f"{summary['hourly_rows']}"
            )
    except (ValueError, FileExistsError, RuntimeError) as error:
        # FileExistsError: prepare refuses to overwrite; show a message, not a traceback.
        parser.error(str(error))
