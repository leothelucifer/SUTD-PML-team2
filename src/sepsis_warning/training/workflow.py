"""Training and other CLI workflows extracted without changing behavior."""

"""Run the sketch end to end, or score an individual patient history."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from ..data import SUBGROUP_COLUMNS
from ..data.cohorts import load_cohorts
from ..data.schema import extreme_value_counts, validate_hourly
from ..evaluation.baselines import (
    BASELINE_THRESHOLD,
    BASELINES,
    auroc,
    average_precision,
    baseline_table,
    hourly_classification_metrics,
    plot_sirs_threshold_tradeoff,
    sirs_threshold_sweep,
)
from ..evaluation.importance import grouped_permutation_importance
from ..evaluation.sirs_failures import failure_tables
from ..features import META, feature_table
from ..models.factory import (
    candidate_models,
    complexity_ranks,
    fit_calibrated,
    is_numpy_model,
    load_model_config,
)
from ..models.policy import apply_policy, calibration_diagnostics, select_candidate, summarize, tune_threshold
from ..models.scoring import file_sha256, load_bundle


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def candidate_details(estimator, seconds) -> dict:
    """What MLflow records per candidate: implementation, settings, fit time, loss curve."""
    final = estimator[-1] if hasattr(estimator, "steps") else estimator
    module = type(final).__module__
    implementation = "numpy" if module.startswith("sepsis_warning") else (
        "reference" if module.startswith("sklearn.dummy") else "sklearn")
    if hasattr(final, "get_params"):
        params = final.get_params(deep=False)
    else:
        params = {key: value for key, value in vars(final).items() if not key.endswith("_")}
    params = {key: value for key, value in params.items() if isinstance(value, (int, float, str, bool, type(None)))}
    history = getattr(final, "loss_history_", None)
    return {"implementation": implementation, "class": type(final).__name__, "params": params,
            "fit_seconds": round(seconds, 3),
            "loss_history": [float(value) for value in history] if history is not None else None}

def check_training_data(args) -> None:
    """Hard schema rules on the whole Parquet table before any model is fitted.

    Also writes the review-flag counts for training stays (small cells hidden).
    """
    if getattr(args, "skip_data_checks", False):
        print("WARNING: data checks skipped (--skip-data-checks)", flush=True)
        return
    try:
        import pandera  # noqa: F401  # pylint: disable=unused-import
    except ImportError as error:
        raise RuntimeError(
            "training on a Parquet table runs the data checks, which need the checks extra: "
            "`pip install -e .[checks]` (or pass --skip-data-checks)"
        ) from error
    frame = pd.read_parquet(args.data)
    validate_hourly(frame)
    print(f"Data checks passed: {frame.groupby(['hospital', 'patient_id']).ngroups} stays, {len(frame)} rows.",
          flush=True)
    splits_path = getattr(args, "splits", None) or args.data.with_name("splits.csv")
    assigned = pd.read_csv(splits_path, dtype=str)
    train = assigned.loc[assigned["split"] == "train", ["hospital", "patient_id"]]
    frame["patient_id"] = frame["patient_id"].astype(str)
    extreme_value_counts(frame.merge(train, on=["hospital", "patient_id"])).to_csv(
        args.output / "extreme_value_review_train.csv", index=False)

def run_experiment(args):
    from ..lifecycle.manifest import build_manifest, write_manifest
    from ..lifecycle.tracking import log_run, run_metadata
    from ..reporting.report import write_report

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"{args.output} is not empty; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    splits, synthetic = load_cohorts(args)
    manifest = []
    for name, cohort in splits.items():
        print(f"{name}: {len(cohort)} patients", flush=True)
        for patient in cohort:
            pid = patient.patient_id.split(":", 1)[1] if ":" in patient.patient_id else patient.patient_id
            if not args.data.is_file():
                path = args.data / f"training_set{patient.hospital}" / f"{pid}.psv"
                sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            else:
                path = args.data.parent / f"training_set{patient.hospital}" / f"{pid}.psv"
                if path.exists():
                    sha = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    sha = hashlib.sha256(patient.patient_id.encode("utf-8")).hexdigest()
            manifest.append({
                "patient_id": patient.patient_id, "hospital": patient.hospital,
                "split": name, "septic": patient.septic,
                "sha256": sha,
            })
    pd.DataFrame(manifest).to_csv(args.output / "split_manifest.csv", index=False)
    if args.data.is_file():
        check_training_data(args)
    # Only training, calibration and validation rows are touched here. The test
    # splits are scored once per frozen model, by the separate `evaluate` command.
    tables = {name: feature_table(splits[name], args.window) for name in ["train", "calibration", "validation"]}
    columns = [c for c in tables["train"] if c not in META]
    model_config, model_config_sha = load_model_config(getattr(args, "models_config", None))
    comparisons = []
    fitted = {}
    details = {}
    calibration_curves = {}
    validation_scores = {}
    candidates = candidate_models(args.seed, model_config, getattr(args, "include_numpy_models", False))
    for name, estimator in candidates.items():
        print(f"Training {name}...", flush=True)
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            model = fit_calibrated(estimator, tables["train"], tables["calibration"], columns, is_constant=name == "constant_prior")
            validation = tables["validation"][META].copy()
            uncalibrated_prob = estimator.predict_proba(tables["validation"][columns])[:, 1]
            validation["probability"] = model.predict_proba(tables["validation"][columns])[:, 1]
        details[name] = candidate_details(estimator, time.perf_counter() - started)
        cal_diag = calibration_diagnostics(validation["label"], uncalibrated_prob, validation["probability"])
        calibration_curves[name] = cal_diag
        metrics = tune_threshold(validation, args.budget, args.cooldown)
        metrics["uncalibrated_brier_score"] = cal_diag["brier_before"]
        comparisons.append({"model": name, **metrics})
        fitted[name] = model
        validation_scores[name] = validation
    selected, selection = select_candidate(comparisons, complexity_ranks(model_config))
    # The rule every model must beat, on the same validation stays and under the same alert
    # budget (its criteria-count threshold is tuned the same way). A promotion gate reads this.
    sirs_table = baseline_table(splits["validation"], "sirs")
    sirs = tune_threshold(sirs_table, args.budget, args.cooldown)
    sirs["brier_score"] = None  # the SIRS score is not a probability
    name, threshold = selected["model"], selected["threshold"]
    pd.DataFrame(comparisons).to_csv(args.output / "validation_models.csv", index=False)
    write_json(args.output / "candidate_details.json", details)
    write_json(args.output / "calibration.json", calibration_curves)
    summary = {
        "data_mode": "synthetic_fixture" if synthetic else "physionet",
        "selected_model": name, "threshold": threshold,
        "validation_alarm_budget": args.budget, "cooldown_hours": args.cooldown,
        "selection": selection, "validation_models": comparisons,
        "calibration": calibration_curves,
        "reference_baselines": {"sirs": sirs},
    }
    config = {
        "seed": args.seed, "window_hours": args.window, "budget": args.budget,
        "cooldown_hours": args.cooldown, "data_root": str(args.data.resolve()),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "synthetic": synthetic, "feature_columns": columns,
        "models": model_config, "models_config_sha256": model_config_sha,
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "metrics.json", summary)
    # The same validation-only summary under the name tracking, S3 and GET /model read.
    write_json(args.output / "selection_metrics.json", summary)
    splits_summary = args.splits_summary
    if splits_summary is None and args.data.is_file():
        splits_path = getattr(args, "splits", None) or args.data.with_name("splits.csv")
        candidate = splits_path.with_name(f"{splits_path.name}.summary.json")
        if candidate.exists():
            try:
                candidate_seed = json.loads(candidate.read_text(encoding="utf-8")).get("seed")
                if candidate_seed is not None and int(candidate_seed) == int(args.seed):
                    splits_summary = candidate
            except (OSError, ValueError, TypeError) as error:
                print(f"WARNING: could not read {candidate} ({error}); the run will not record split hashes",
                      flush=True)
    metadata = run_metadata(
        args.seed, synthetic=synthetic, window=args.window, budget=args.budget,
        cooldown=args.cooldown, feature_columns=columns,
        splits_summary=splits_summary, dvc_lock=args.dvc_lock,
    )
    metadata["models_config_sha256"] = model_config_sha
    write_json(args.output / "run_metadata.json", metadata)
    manifest = build_manifest(
        model_name=name, threshold=threshold, cooldown=args.cooldown, window=args.window,
        synthetic=synthetic, feature_columns=columns, metadata=metadata, validation=selected,
        models_config_sha256=model_config_sha, numpy_model=is_numpy_model(fitted[name]),
    )
    bundle = {
        "model": fitted[name], "model_name": name, "feature_columns": columns,
        "window": args.window, "threshold": threshold, "cooldown": args.cooldown,
        "synthetic": synthetic, "manifest": manifest,
    }
    joblib.dump(bundle, args.output / "model.joblib")
    write_manifest(args.output, manifest, file_sha256(args.output / "model.joblib"))
    # Which measurements the selected model relies on: validation rows only, nothing is refitted.
    importance_repeats = getattr(args, "importance_repeats", 3)
    if importance_repeats > 0:
        print(f"Permutation importance of {name} on validation ({importance_repeats} shuffle(s) per group)...",
              flush=True)
        with threadpool_limits(limits=2):
            grouped_permutation_importance(
                fitted[name], tables["validation"], columns, repeats=importance_repeats, seed=args.seed,
            ).to_csv(args.output / "feature_importance.csv", index=False)
    # Every candidate's validation scores, plus the SIRS score, so `model-report` can compare all models
    # and the baseline (precision, recall, F1, PR curves) without refitting. Validation stays only.
    # Patient-level, so it stays in the run folder (git-ignored) and is never logged to MLflow.
    all_scores = validation_scores[name][META].copy()
    # Recorded age and gender of each stay, for the subgroup table (bias audit) in `model-report`.
    for attribute in SUBGROUP_COLUMNS:
        all_scores[attribute] = tables["validation"][attribute].to_numpy()
    for candidate, table in validation_scores.items():
        all_scores[f"p_{candidate}"] = table["probability"].to_numpy()
    all_scores = all_scores.merge(
        sirs_table[["patient_id", "hour", "probability"]].rename(columns={"probability": "sirs_score"}),
        on=["patient_id", "hour"], how="left", validate="one_to_one")
    all_scores.to_csv(args.output / "validation_scores.csv", index=False)
    # Replay of validation stays only: these were already used for selection.
    replay = apply_policy(validation_scores[name], threshold, args.cooldown)
    replay["split"] = "validation"
    replay.to_csv(args.output / "validation_predictions.csv", index=False)
    write_report(args.output / "replay.html", replay,
                 {**summary, "splits": {"validation": selected}}, threshold, synthetic)
    print(f"Selected: {name}; threshold={threshold:.2f} "
          f"(best by utility: {selection['best_by_utility']}; within one standard error: "
          f"{', '.join(selection['within_one_standard_error'])})", flush=True)
    print(f"Validation replay: {(args.output / 'replay.html').resolve()}", flush=True)
    print("Test splits were not scored. After the model is reviewed and frozen, run `evaluate` once.", flush=True)
    if args.mlflow_uri:
        run_id = log_run(args.output, args.mlflow_uri)
        print(f"MLflow run: {run_id}", flush=True)
        if args.register_model:
            from ..lifecycle.registry import MODEL_NAME, register_bundle

            if getattr(args, "require_gates", False):
                # Pipeline use: a model that cannot be promoted is not registered at all.
                from ..lifecycle.gates import require_gates

                require_gates(summary, require_real_data=getattr(args, "require_real_data", False))

            version = register_bundle(args.output, args.mlflow_uri, run_id)
            print(f"Registered {MODEL_NAME} version {version} as @candidate "
                  f"(models:/{MODEL_NAME}@candidate). Promote only after review.", flush=True)

def run_tune(args):
    """Hyperparameter search on the training split only; writes tuned settings for `train --models-config`."""
    from ..lifecycle.tracking import log_tuning
    from . import tuning

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"{args.output} is not empty; use a new output directory")
    cohorts, synthetic = load_cohorts(args)
    models_config, models_sha = load_model_config(args.models_config)
    space, space_sha = tuning.load_space(args.space_config)
    results, summary, tuned = tuning.run_search(
        cohorts["train"], models_config=models_config, space=space, window=args.window, folds=args.folds,
        trials=args.trials, seed=args.seed, max_stays=args.max_stays, only=args.models,
        progress=lambda message: print(message, flush=True),
    )
    data_note = "synthetic_fixture" if synthetic else "physionet"
    tuning.write_outputs(args.output, results, summary, tuned, models_config_sha256=models_sha,
                         space_sha256=space_sha, data_note=data_note)
    print("\nBest configuration per model (cross-validated average precision, training split only):")
    for name, best in summary["models"].items():
        marker = "kept the reference settings" if best["winner_is_reference"] else f"gain {best['ap_gain_over_reference']:+.4f}"
        print(f"  {name}: AP {best['cv_ap_mean']:.4f} (reference {best['reference_cv_ap_mean']:.4f}; {marker})")
    print(f"Tuned settings: {(args.output / 'tuned_models.toml').resolve()}")
    print("Next: python -m sepsis_warning train --models-config "
          f"{args.output / 'tuned_models.toml'} ...", flush=True)
    if args.mlflow_uri:
        print(f"MLflow tuning run: {log_tuning(args.output, args.mlflow_uri)}", flush=True)

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
    # The saved split and a seed-42 re-split are the same stays, so rows line up with `train`.
    splits, _ = load_cohorts(args)
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
        f"python -m sepsis_warning baseline --data {args.data.as_posix()} --output artifacts/baseline",
    )
    print(sweep[["alert_at_criteria", "hourly_precision", "hourly_recall", "hourly_f1",
                 "hourly_specificity", "false_alert_events_per_nonseptic_patient_day"]].round(3).to_string(index=False))
    print(f"Threshold sweep (validation): {sweep_path.resolve()}", flush=True)
    return result

def run_server(args):
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_non_local_host:
        raise ValueError(f"--host {args.host} exposes an unauthenticated API; pass --allow-non-local-host if agreed")
    try:
        import uvicorn

        from ..serving.api import create_app
    except ImportError as error:
        raise RuntimeError("the API needs the serve extra; run `pip install -e .[serve]`") from error
    from ..lifecycle.model_source import load_serving_model
    from ..serving.service import ModelMismatch

    try:
        alert_store = None
        if args.alert_table:
            from ..serving.alert_store import DynamoAlertStore

            alert_store = DynamoAlertStore(args.alert_table)
        elif args.alert_ledger == "memory":
            from ..serving.alert_store import InMemoryAlertStore

            alert_store = InMemoryAlertStore()
        service, card, source = load_serving_model(
            model=args.model, model_uri=args.model_uri, model_s3=args.model_s3,
            expected_sha256=args.model_sha256, mlflow_uri=args.mlflow_uri,
            allow_unverified=args.allow_unverified_model, alert_store=alert_store,
        )
    except ModelMismatch as error:
        raise ValueError(str(error)) from error
    print(f"Loaded {source}", flush=True)
    health = service.health()
    print(f"Serving {health['model_name']} (bundle {health['bundle_sha256'][:12]}) on {args.host}:{args.port}", flush=True)
    if card is None:
        print("No selection files next to the bundle; GET /model will not show the selection", flush=True)
    if health["synthetic_model"]:
        print("WARNING: synthetic fixture model; scores are meaningless", flush=True)
    telemetry = None
    if args.telemetry or args.prediction_log:
        from ..monitoring.telemetry import Telemetry

        telemetry = Telemetry(prediction_log=args.prediction_log, environment=args.environment)
    uvicorn.run(create_app(service, card, telemetry), host=args.host, port=args.port, workers=1,
                access_log=telemetry is None)

def run_sirs_failures(args):
    """Aggregate SIRS failure patterns. Validation split only, by design."""
    cohorts, _ = load_cohorts(args)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in failure_tables(cohorts["validation"], args.cooldown).items():
        table.to_csv(args.output / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.round(3).to_string(index=False)}", flush=True)
    print(f"\nTables: {args.output.resolve()}", flush=True)

def run_validate(args):
    """Check every stay against the hard rules, then count review flags on train only."""
    for path, step in [(args.data, "prepare"), (args.splits, "split")]:
        if not path.exists():
            raise ValueError(f"{path} not found; run `{step}` first")
    frame = pd.read_parquet(args.data)
    validate_hourly(frame)
    print(f"All {frame.groupby(['hospital', 'patient_id']).ngroups} stays and {len(frame)} rows pass the hard rules.", flush=True)
    splits = pd.read_csv(args.splits, dtype=str)
    train = splits.loc[splits["split"] == "train", ["hospital", "patient_id"]]
    table = extreme_value_counts(frame.merge(train, on=["hospital", "patient_id"]))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "extreme_value_review_train.csv"
    table.to_csv(path, index=False)
    print(f"\nReview flags, training stays only (nothing rejected):\n{table.to_string(index=False)}")
    print(f"\nReview table: {path.resolve()}", flush=True)

def run_ward(args):
    from ..reporting.ward import build_payload, check_split, choose_stays, score_locally, score_via_api, write_ward
    """Write the ICU ward replay page for a saved model (see ward.py)."""
    check_split(args.split, args.after_freeze)
    if args.output.exists():
        raise ValueError(f"{args.output} already exists; choose a new output file")
    cohorts, synthetic = load_cohorts(args)
    cohort = cohorts[args.split]
    stays = choose_stays(cohort, args.beds, args.septic_share, args.max_hours, args.seed)
    if args.api_url:
        frames, details = score_via_api(args.api_url, stays, sign=args.sign)
        if bool(details["synthetic_model"]) != synthetic:
            raise ValueError("The API's model and the data disagree on whether they are synthetic")
        model_name, threshold, cooldown = details["model_name"], details["threshold"], details["cooldown_hours"]
        scored_by = f"POST /score on {args.api_url}, one request per bed, as the hospital engine would send"
    else:
        bundle = load_bundle(args.model)
        if bool(bundle["synthetic"]) != synthetic:
            raise ValueError("The model bundle and the data disagree on whether they are synthetic")
        frames = score_locally(bundle, stays)
        model_name, threshold, cooldown = bundle["model_name"], bundle["threshold"], bundle["cooldown"]
        scored_by = "Saved model scored in-process, with the same feature code as the API"
    payload = build_payload(
        frames, model_name=model_name, threshold=threshold, cooldown=cooldown, synthetic=synthetic,
        split=args.split, scored_by=scored_by, seed=args.seed,
        cohort_septic_share=sum(p.septic for p in cohort) / len(cohort),
    )
    write_ward(args.output, payload)
    if not synthetic:
        print("Contains PhysioNet stays: keep it private and out of Git.", flush=True)
    print(f"Ward replay: {args.output.resolve()}", flush=True)
