"""Label-free drift checks on the live prediction log (the fast checks of ``monitoring.py``).

Input: prediction lines written by ``telemetry.py`` (hashed stay ID, hour,
probability, alert flag, which measurements were missing). They hold no
measurement values, so live checks cover what can be seen without them:

* **score shift**: PSI of the probabilities against the frozen reference;
* **alert rate**: new alerts per patient-day, as a ratio to the reference;
* **missingness shift**: per measurement, how much more or less often it is
  missing than in the reference (a changed lab panel or a broken feed).

Value-distribution drift and performance drift still need the offline replay
(``monitor-replay``) on stays with settled labels.

The reference is the file ``monitor-setup`` freezes for one bundle. A log
produced by another bundle is reported as ``reference_mismatch`` instead of
being compared. An alarm asks a person to look; nothing here retrains or
changes a threshold.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import DYNAMIC
from .offline import RATE_EPSILON, bin_shares, psi
from .prediction_log import read_prediction_log  # noqa: F401  (re-exported: the drift job and the tests import it from here)

MIN_ROWS = 200
THRESHOLDS: dict[str, dict] = {
    "score_psi": {"warn": 0.10, "alarm": 0.25},
    # Alerts per patient-day relative to the reference: half or double is a warning.
    "alert_rate_ratio": {"warn": (0.5, 2.0), "alarm": (0.25, 4.0)},
    "max_missingness_shift": {"warn": 0.10, "alarm": 0.20},
}
# 0 ok, 1 warn, 2 alarm. A log from another model than the reference cannot be judged at all, which
# is worse than a drifting score, so it also reaches the alarm line (level 3). Too little data is -1.
LEVEL = {"ok": 0, "warn": 1, "alarm": 2, "reference_mismatch": 3, "insufficient_data": -1}


def _level(value, warn, alarm) -> str:
    """ok / warn / alarm for a value against one-sided limits or (low, high) bands."""
    if value is None:
        return "ok"
    for level, bound in (("alarm", alarm), ("warn", warn)):
        if bound is None:
            continue
        outside = not (bound[0] <= value <= bound[1]) if isinstance(bound, tuple) else value > bound
        if outside:
            return level
    return "ok"


def _status(name: str, value) -> str:
    limits: dict = THRESHOLDS[name]
    return _level(value, limits["warn"], limits["alarm"])


def _frozen(reference: dict, key: str) -> dict | None:
    """Warn/alert limits that monitor-setup froze with the reference (null-window quantiles above the floors)."""
    entry = (reference.get("thresholds") or {}).get(key)
    if isinstance(entry, dict) and entry.get("warn") is not None:
        return {"warn": float(entry["warn"]), "alarm": None if entry.get("alert") is None else float(entry["alert"])}
    return None


def live_checks(predictions: pd.DataFrame, reference: dict) -> dict:
    """Compare a window of prediction records with the frozen reference."""
    rows = len(predictions)
    checks: dict[str, dict] = {}
    result: dict = {"rows": rows, "stays": int(predictions["stay"].nunique()) if rows else 0, "checks": checks}
    if rows:
        bundles = sorted(set(predictions["bundle_sha256"].dropna().astype(str)))
        result["bundles"] = bundles
        expected = str(reference.get("model_sha256", ""))[:12]
        if expected and any(b != expected for b in bundles):
            result["status"] = "reference_mismatch"
            result["detail"] = (f"the log holds bundle(s) {bundles}, but the reference was frozen for {expected}; "
                                "run monitor-setup for the deployed model")
            return result
    if rows < MIN_ROWS:
        result["status"] = "insufficient_data"
        result["detail"] = f"{rows} prediction rows; at least {MIN_ROWS} are needed"
        return result

    score = reference["score"]
    live_shares = bin_shares(predictions["probability"].to_numpy(dtype=float), np.asarray(score["edges"]))
    score_psi = psi(score["shares"], live_shares)
    live_rate = float(predictions["new_alert"].astype(bool).sum() / (rows / 24))
    reference_rate = score.get("alerts_per_patient_day") or 0.0
    ratio = live_rate / reference_rate if reference_rate > 0 else None

    missing_counts = pd.Series(0, index=DYNAMIC, dtype=float)
    for names in predictions["missing_now"]:
        missing_counts.loc[[n for n in names if n in missing_counts.index]] += 1
    shifts = {}
    for name in DYNAMIC:
        baseline = (reference.get("inputs", {}).get(name) or {}).get("missing_fraction")
        if baseline is not None:
            shifts[name] = float(missing_counts[name] / rows - baseline)
    worst = sorted(shifts.items(), key=lambda item: -abs(item[1]))[:5]
    max_shift = abs(worst[0][1]) if worst else None

    # Limits come from the reference frozen with the model (monitor-setup) where it has them; the constants
    # in THRESHOLDS are the fallback for an older reference file. The source is reported with the result.
    sources = {}
    psi_limits = _frozen(reference, "score_psi")
    sources["score_psi"] = "reference" if psi_limits else "defaults"
    psi_limits = psi_limits or THRESHOLDS["score_psi"]
    checks["score_psi"] = {"value": score_psi, "status": _level(score_psi, psi_limits["warn"], psi_limits["alarm"]),
                           "thresholds": psi_limits}

    rate_limits = _frozen(reference, "alert_rate_log_ratio")
    if rate_limits and reference_rate >= 0:
        log_ratio = abs(math.log((live_rate + RATE_EPSILON) / (reference_rate + RATE_EPSILON)))
        rate_status = _level(log_ratio, rate_limits["warn"], rate_limits["alarm"])
        sources["alert_rate_ratio"] = "reference"
        rate_thresholds = {**rate_limits, "statistic": "abs(ln(live/reference))"}
    else:
        rate_status = _status("alert_rate_ratio", ratio)
        sources["alert_rate_ratio"] = "defaults"
        rate_thresholds = THRESHOLDS["alert_rate_ratio"]
    checks["alert_rate_ratio"] = {"value": ratio, "status": rate_status, "thresholds": rate_thresholds}

    miss_status, miss_sources = "ok", set()
    for name, shift in shifts.items():
        limits = _frozen(reference, f"missing:{name}")
        miss_sources.add("reference" if limits else "defaults")
        limits = limits or THRESHOLDS["max_missingness_shift"]
        level = _level(abs(shift), limits["warn"], limits["alarm"])
        if LEVEL[level] > LEVEL[miss_status]:
            miss_status = level
    sources["max_missingness_shift"] = "reference" if miss_sources == {"reference"} else (
        "defaults" if miss_sources == {"defaults"} else "mixed")
    checks["max_missingness_shift"] = {"value": max_shift, "status": miss_status,
                                       "thresholds": "per variable from the reference" if "reference" in miss_sources
                                       else THRESHOLDS["max_missingness_shift"]}
    result["threshold_source"] = sources
    result["alerts_per_patient_day"] = {"live": live_rate, "reference": reference_rate}
    result["largest_missingness_shifts"] = [{"input": name, "shift": shift} for name, shift in worst]
    result["status"] = max((c["status"] for c in checks.values()), key=lambda status: LEVEL[status])
    return result


def run_live_drift(sources, reference_path: Path, output: Path | None = None) -> dict:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    result = live_checks(read_prediction_log(sources), reference)
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "live_drift.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                      encoding="utf-8")
    return result


def emf_record(result: dict, environment: str) -> dict:
    """CloudWatch Embedded Metric Format line for a drift result (printed by the scheduled job)."""
    checks = result.get("checks", {})
    record = {
        "_aws": {"Timestamp": int(pd.Timestamp.now(tz="UTC").timestamp() * 1000), "CloudWatchMetrics": [{
            "Namespace": "SepsisWarning/Drift", "Dimensions": [["Environment"]],
            "Metrics": [{"Name": "DriftLevel", "Unit": "None"}, {"Name": "ScorePSI", "Unit": "None"},
                        {"Name": "MaxMissingnessShift", "Unit": "None"}, {"Name": "PredictionRows", "Unit": "Count"}],
        }]},
        "record_type": "drift", "Environment": environment, "status": result["status"],
        # 0 ok, 1 warn, 2 alarm, 3 wrong reference (also alarms); -1 when there was too little data.
        "DriftLevel": LEVEL.get(result["status"], -1),
        "ScorePSI": (checks.get("score_psi") or {}).get("value") or 0.0,
        "MaxMissingnessShift": (checks.get("max_missingness_shift") or {}).get("value") or 0.0,
        "PredictionRows": result["rows"], "detail": result.get("detail"),
    }
    return record
