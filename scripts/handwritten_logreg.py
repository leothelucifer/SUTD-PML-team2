"""Train the hand-written NumPy logistic regression and report statistics.

    python scripts/handwritten_logreg.py --data data/physionet/combined.parquet \
        --splits data/physionet/splits.csv --output artifacts/handwritten

Fits on train, calibrates on the calibration split, and evaluates and sets the
alert threshold on validation only. The test splits are never read. Outputs
are aggregates (no patient rows). scikit-learn is used only as an independent
check that the coefficients match the same objective.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from sepsis_warning.baselines import auroc, average_precision
from sepsis_warning.features import META, feature_table
from sepsis_warning.handwritten import CalibratedNumpyModel, NumpyLogisticRegression, PlattCalibrator
from sepsis_warning.policy import apply_policy, summarize, tune_threshold
from sepsis_warning.splits import load_split_patients


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_bins(y, p, bins=5):
    """Equal-count bins of predicted risk; mean predicted vs observed rate."""
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({
            "hours": int(len(chunk)),
            "mean_predicted": float(np.mean(p[chunk])),
            "observed_rate": float(np.mean(y[chunk])),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/physionet/combined.parquet"))
    parser.add_argument("--splits", type=Path, default=Path("data/physionet/splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/handwritten"))
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--C", type=float, default=0.3)
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--cooldown", type=int, default=6)
    args = parser.parse_args()

    cohorts = load_split_patients(args.data, args.splits)
    tables = {name: feature_table(cohorts[name], args.window) for name in ("train", "calibration", "validation")}
    columns = [c for c in tables["train"] if c not in META]
    X = {name: table[columns].to_numpy(dtype=np.float64) for name, table in tables.items()}
    y = {name: table["label"].to_numpy(dtype=int) for name, table in tables.items()}

    started = time.perf_counter()
    model = NumpyLogisticRegression(C=args.C).fit(X["train"], y["train"])
    fit_seconds = time.perf_counter() - started
    calibrator = PlattCalibrator().fit(model.decision_function(X["calibration"]), y["calibration"])
    calibrated = CalibratedNumpyModel(model, calibrator)

    raw_val = model.predict_proba(X["validation"])[:, 1]
    cal_val = calibrated.predict_proba(X["validation"])[:, 1]
    validation = tables["validation"][META].copy()
    validation["probability"] = cal_val
    policy = tune_threshold(validation, args.budget, args.cooldown)
    at_threshold = apply_policy(validation, policy["threshold"], args.cooldown)
    decided = at_threshold["hourly_positive"].to_numpy().astype(bool)
    labels = y["validation"].astype(bool)

    # Independent check: scikit-learn on the same scaled inputs and objective.
    reference = LogisticRegression(C=args.C, class_weight="balanced", solver="lbfgs", max_iter=5000, tol=1e-10)
    reference.fit(model.preprocessor.transform(X["train"]), y["train"])
    coef_gap = float(np.max(np.abs(reference.coef_[0] - model.coef_)))

    top = np.argsort(-np.abs(model.coef_))[:10]
    summary = {
        "data": {name: {"stays": len(cohorts[name]), "septic_stays": int(sum(p.septic for p in cohorts[name])),
                        "hours": int(len(y[name])), "positive_hours": int(y[name].sum())}
                 for name in ("train", "calibration", "validation")},
        "features": len(columns),
        "fit": {"C": args.C, "newton_iterations": len(model.history) - 1, "converged": model.converged,
                "objective_start": model.history[0], "objective_end": model.history[-1],
                "seconds": round(fit_seconds, 3)},
        "matches_sklearn": {"max_abs_coefficient_difference": coef_gap,
                            "intercept_difference": float(abs(reference.intercept_[0] - model.intercept_))},
        "platt": {"a": calibrator.a, "b": calibrator.b},
        "train": {"auroc": auroc(y["train"], model.predict_proba(X["train"])[:, 1]),
                  "average_precision": average_precision(y["train"], model.predict_proba(X["train"])[:, 1])},
        "validation": {
            "auroc": auroc(y["validation"], cal_val),
            "average_precision": average_precision(y["validation"], cal_val),
            "brier_before_calibration": brier(y["validation"], raw_val),
            "brier_after_calibration": brier(y["validation"], cal_val),
            "mean_predicted_risk": float(cal_val.mean()),
            "positive_hour_rate": float(y["validation"].mean()),
            "calibration_bins": calibration_bins(y["validation"], cal_val),
        },
        "alert_policy": {k: policy[k] for k in (
            "threshold", "normalized_hourly_utility", "alert_events", "false_alert_events_per_nonseptic_patient_day",
            "early_detection_recall", "eligible_septic_patients_for_lead_time", "median_inferred_lead_hours")},
        "hourly_confusion_at_threshold": {
            "tp": int((labels & decided).sum()), "fp": int((~labels & decided).sum()),
            "fn": int((labels & ~decided).sum()), "tn": int((~labels & ~decided).sum()),
        },
        "top_coefficients": [{"feature": columns[i], "coefficient": float(model.coef_[i])} for i in top],
        "baseline_validation": summarize(apply_policy(
            validation.assign(probability=float(y["train"].mean())), 0.5, args.cooldown))["normalized_hourly_utility"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
