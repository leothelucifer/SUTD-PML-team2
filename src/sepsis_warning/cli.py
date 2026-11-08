"""Run the sketch end to end, or score an individual patient history."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .baselines import (
    BASELINE_THRESHOLD, BASELINES, auroc, average_precision, baseline_table,
    hourly_classification_metrics, plot_sirs_threshold_tradeoff, sirs_threshold_sweep,
)
from .data import generate_synthetic, load_patients, read_patient, split_patients
from .hospital_comparison import run_comparison
from .features import META, build_features, feature_table, select_saved_features
from .modeling import candidate_models, fit_calibrated
from .missingness import run_missingness_analysis
from .policy import apply_policy, summarize, tune_threshold
from .prepare import prepare_parquet, write_summary
from .report import write_report
from .splits import write_splits


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"{args.output} is not empty; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    synthetic = (args.data / "SYNTHETIC.json").exists()
    print("Data mode:", "SYNTHETIC SOFTWARE FIXTURE" if synthetic else "PhysioNet records", flush=True)
    patients = load_patients(args.data)
    if not synthetic and any("synthetic_" in p.patient_id for p in patients):
        raise ValueError("Synthetic filenames found without SYNTHETIC.json marker")
    splits = split_patients(patients, args.seed)
    manifest = []
    for name, cohort in splits.items():
        print(f"{name}: {len(cohort)} patients", flush=True)
        for patient in cohort:
            path = args.data / f"training_set{patient.hospital}" / f"{patient.patient_id.split(':', 1)[1]}.psv"
            manifest.append({
                "patient_id": patient.patient_id, "hospital": patient.hospital,
                "split": name, "septic": patient.septic,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    pd.DataFrame(manifest).to_csv(args.output / "split_manifest.csv", index=False)
    # Only source training/calibration/validation features are built before
    # selecting the model and threshold. Test outcomes never guide selection.
    tables = {name: feature_table(splits[name], args.window) for name in ["train", "calibration", "validation"]}
    columns = [c for c in tables["train"] if c not in META]
    comparisons = []
    fitted = {}
    for name, estimator in candidate_models(args.seed).items():
        print(f"Training {name}...", flush=True)
        with threadpool_limits(limits=2):
            model = fit_calibrated(estimator, tables["train"], tables["calibration"], columns, is_constant=name == "constant_prior")
            validation = tables["validation"][META].copy()
            validation["probability"] = model.predict_proba(tables["validation"][columns])[:, 1]
        metrics = tune_threshold(validation, args.budget, args.cooldown)
        comparisons.append({"model": name, **metrics})
        fitted[name] = model
    selected = max(comparisons, key=lambda m: (
        m["normalized_hourly_utility"], -m["false_alert_events_per_nonseptic_patient_day"],
        m["average_precision"],
    ))
    name, threshold = selected["model"], selected["threshold"]
    pd.DataFrame(comparisons).to_csv(args.output / "validation_models.csv", index=False)
    bundle = {
        "model": fitted[name], "model_name": name, "feature_columns": columns,
        "window": args.window, "threshold": threshold, "cooldown": args.cooldown,
        "synthetic": synthetic,
    }
    joblib.dump(bundle, args.output / "model.joblib")
    held_out = {}
    predictions = []
    for split in ["internal_test", "external_test"]:
        table = feature_table(splits[split], args.window)
        predicted = table[META].copy()
        with threadpool_limits(limits=2):
            predicted["probability"] = fitted[name].predict_proba(table[columns])[:, 1]
        predicted = apply_policy(predicted, threshold, args.cooldown)
        held_out[split] = summarize(predicted)
        predicted["split"] = split
        predictions.append(predicted)
    summary = {
        "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "selected_model": name, "threshold": threshold,
        "validation_alarm_budget": args.budget, "cooldown_hours": args.cooldown,
        "validation_models": comparisons, "held_out": held_out,
    }
    config = {
        "seed": args.seed, "window_hours": args.window, "budget": args.budget,
        "cooldown_hours": args.cooldown, "data_root": str(args.data.resolve()),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "synthetic": synthetic, "feature_columns": columns,
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "metrics.json", summary)
    replay = pd.concat(predictions, ignore_index=True)
    replay.to_csv(args.output / "held_out_predictions.csv", index=False)
    write_report(args.output / "replay.html", replay, summary, threshold, synthetic)
    print(f"Selected: {name}; threshold={threshold:.2f}", flush=True)
    print(f"Replay: {(args.output / 'replay.html').resolve()}", flush=True)


BASELINE_SPLITS = ["validation", "internal_test", "external_test"]
BASELINE_COLUMNS = [
    "model", "split", "patients", "septic_patients", "hours", "positive_hour_fraction",
    "hourly_precision", "hourly_recall", "hourly_f1", "hourly_specificity", "hourly_accuracy",
    "hourly_alert_rate", "no_predicted_positives",
    "normalized_hourly_utility", "alert_events",
    "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
    "median_inferred_lead_hours", "average_precision", "auroc",
    "eligible_septic_patients_for_lead_time", "left_censored_septic_patients",
    "hourly_tp", "hourly_fp", "hourly_fn", "hourly_tn",
]
SWEEP_COLUMNS = [
    "alert_at_criteria", "hourly_precision", "hourly_recall", "hourly_f1", "hourly_specificity",
    "hourly_accuracy", "hourly_alert_rate", "normalized_hourly_utility",
    "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
    "hourly_tp", "hourly_fp", "hourly_fn", "hourly_tn",
]


def run_baselines(args):
    """Score SIRS and always-no on validation, internal test and hospital B."""
    synthetic = (args.data / "SYNTHETIC.json").exists()
    print("Data mode:", "SYNTHETIC SOFTWARE FIXTURE" if synthetic else "PhysioNet records", flush=True)
    patients = load_patients(args.data)
    if not synthetic and any("synthetic_" in p.patient_id for p in patients):
        raise ValueError("Synthetic filenames found without SYNTHETIC.json marker")
    # Same seed and split function as `train`, so rows line up with the models.
    splits = split_patients(patients, args.seed)
    rows = []
    for split in BASELINE_SPLITS:
        for name in BASELINES:
            table = apply_policy(baseline_table(splits[split], name), BASELINE_THRESHOLD, args.cooldown)
            metrics = summarize(table)
            metrics.update(hourly_classification_metrics(table["label"], table["hourly_positive"]))
            # Hand-written ranking metrics replace the sklearn ones from summarize.
            metrics["auroc"] = auroc(table["label"], table["probability"])
            metrics["average_precision"] = average_precision(table["label"], table["probability"])
            rows.append({"model": name, "split": split, **metrics})
    result = pd.DataFrame(rows)[BASELINE_COLUMNS]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "baseline_metrics.csv"
    result.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(result[BASELINE_COLUMNS[:2] + [
            "hourly_precision", "hourly_recall", "hourly_f1", "normalized_hourly_utility",
            "false_alert_events_per_nonseptic_patient_day", "early_detection_recall",
            "auroc", "average_precision",
        ]].round(3).to_string(index=False))
    print(f"Baseline metrics: {path.resolve()}", flush=True)
    # Threshold trade-off (course L4) on VALIDATION ONLY. Descriptive: the rule
    # stays at >= 2 criteria; test splits are never used to look at thresholds.
    validation = baseline_table(splits["validation"], "sirs")
    sweep = sirs_threshold_sweep(validation, args.cooldown)[SWEEP_COLUMNS]
    sweep_path = args.output / "sirs_threshold_sweep_validation.csv"
    sweep.to_csv(sweep_path, index=False)
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_sirs_threshold_tradeoff(
        sweep, float(validation["label"].mean()), 0.2, args.figures / "08_sirs_threshold_tradeoff.png",
        "python -m sepsis_warning baseline --data data/physionet --output artifacts/baseline",
    )
    print(sweep[["alert_at_criteria", "hourly_precision", "hourly_recall", "hourly_f1",
                 "hourly_specificity", "false_alert_events_per_nonseptic_patient_day"]].round(3).to_string(index=False))
    print(f"Threshold sweep (validation): {sweep_path.resolve()}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate an explicitly synthetic fixture and run it")
    demo.add_argument("--data", type=Path, default=Path("data/synthetic"))
    for command in [demo, sub.add_parser("train", help="Run on downloaded training_setA/B")]:
        if command is not demo:
            command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, default=Path("artifacts/demo" if command is demo else "artifacts/real"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--window", type=int, default=6)
        command.add_argument("--budget", type=float, default=0.2)
        command.add_argument("--cooldown", type=int, default=6)
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
    baseline.add_argument("--data", type=Path, required=True)
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--cooldown", type=int, default=6)
    baseline.add_argument("--figures", type=Path, default=Path("reports/figures"))
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
    args = parser.parse_args()
    try:
        if args.command == "demo":
            generate_synthetic(args.data, args.seed)
            write_json(args.data / "SYNTHETIC.json", {"purpose": "Software test only", "seed": args.seed})
        if args.command in ["demo", "train"]:
            if args.window < 1 or args.budget < 0 or args.cooldown < 0:
                raise ValueError("window must be positive; budget and cooldown must be nonnegative")
            run_experiment(args)
        elif args.command == "replay":
            bundle = joblib.load(args.model)
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
        elif args.command == "baseline":
            if args.cooldown < 0:
                raise ValueError("cooldown must be nonnegative")
            run_baselines(args)
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
    except (ValueError, FileExistsError) as error:
        # FileExistsError: prepare refuses to overwrite; show a message, not a traceback.
        parser.error(str(error))
