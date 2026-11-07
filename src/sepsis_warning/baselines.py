"""Rule-based reference baselines that every trained model must beat.

Neither baseline is trained, so neither uses any training data:

* ``sirs``      - the classic SIRS screen (>= 2 of 4 criteria), applied hourly.
* ``always_no`` - never alerts. Shows what "doing nothing" scores.

Both produce the same table format as the models (``patient_id``, ``hospital``,
``hour``, ``label``, ``probability``) so they go through the exact same
``policy.apply_policy`` (threshold + cooldown) and ``policy.summarize`` code.
"""

import numpy as np
import pandas as pd

from .data import Patient

# Each criterion is one or more (column, test) pairs joined by OR.
# Units follow the PhysioNet 2019 data: Temp in deg C, HR in bpm, Resp in
# breaths/min, PaCO2 in mmHg, WBC in 10^9/L. Band neutrophils (> 10%) are part
# of the clinical WBC criterion but are not recorded in this dataset.
SIRS_CRITERIA = {
    "sirs_temp": [("Temp", lambda v: (v > 38) | (v < 36))],
    "sirs_hr": [("HR", lambda v: v > 90)],
    "sirs_resp": [("Resp", lambda v: v > 20), ("PaCO2", lambda v: v < 32)],
    "sirs_wbc": [("WBC", lambda v: (v > 12) | (v < 4))],
}
SIRS_COLUMNS = sorted({column for tests in SIRS_CRITERIA.values() for column, _ in tests})
SIRS_ALERT_COUNT = 2
BASELINES = ["sirs", "always_no"]


def sirs_criteria(frame: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per SIRS criterion, for each hour of ONE patient.

    Causal: each hour uses the last value observed at or before that hour
    (forward fill only, never backward fill). A variable that has never been
    observed is treated as normal, because NaN comparisons evaluate to False.
    """
    return sirs_criteria_from_last_seen(frame[SIRS_COLUMNS].ffill())


def sirs_criteria_from_last_seen(last_seen: pd.DataFrame) -> pd.DataFrame:
    """Apply the SIRS thresholds to values that are ALREADY forward-filled.

    Use this for many patients at once after a per-patient
    ``groupby(...).ffill()``, so values never cross from one stay to the next.
    """
    result = pd.DataFrame(index=last_seen.index)
    for name, tests in SIRS_CRITERIA.items():
        met = np.zeros(len(last_seen), dtype=bool)
        for column, test in tests:
            met |= test(last_seen[column]).to_numpy(dtype=bool)
        result[name] = met
    return result


def sirs_count(frame: pd.DataFrame) -> np.ndarray:
    """Number of SIRS criteria met (0-4) at each hour of one patient."""
    return sirs_criteria(frame).sum(axis=1).to_numpy(dtype=int)


def baseline_table(patients: list[Patient], baseline: str) -> pd.DataFrame:
    """Hourly scores in the same layout the trained models produce.

    ``probability`` is a score, not a calibrated probability:
    * sirs: criteria met / 4, so the 0.5 threshold means ">= 2 criteria".
    * always_no: 0 everywhere, so the 0.5 threshold never fires.
    Ranking metrics (AP, AUROC) on the SIRS score are still meaningful;
    the Brier score is not, and should not be reported for baselines.
    """
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline!r}; choose from {BASELINES}")
    chunks = []
    for patient in patients:
        frame = patient.frame
        if baseline == "sirs":
            score = sirs_count(frame) / len(SIRS_CRITERIA)
        else:
            score = np.zeros(len(frame))
        chunks.append(pd.DataFrame({
            "patient_id": patient.patient_id,
            "hospital": patient.hospital,
            "hour": frame["ICULOS"].to_numpy(),
            "label": frame["SepsisLabel"].to_numpy(dtype=int),
            "probability": score,
        }))
    return pd.concat(chunks, ignore_index=True)


# Threshold applied to ``probability`` by policy.apply_policy.
BASELINE_THRESHOLD = SIRS_ALERT_COUNT / len(SIRS_CRITERIA)


def auroc(labels, scores):
    """Area under the ROC curve via the Mann-Whitney rank formula.

    = probability a random positive hour scores above a random negative hour
    (ties count half). Tied scores get their average rank.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    # Average rank (1-based) within each block of equal scores.
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels, scores):
    """AP = sum over thresholds of (recall_k - recall_{k-1}) * precision_k.

    Thresholds are the distinct scores, highest first, so tied hours enter
    together (same convention as scikit-learn).
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    if not n_pos:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # Last index of each block of equal scores.
    block_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_labels)[block_ends]
    flagged = block_ends + 1
    precision = tp / flagged
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def hourly_classification_metrics(labels, decisions) -> dict:
    """Hand-written hourly precision, recall and F1 (no library metrics).

    Precision is undefined (None) when nothing is flagged, e.g. always-no.
    """
    labels = np.asarray(labels, dtype=bool)
    decisions = np.asarray(decisions, dtype=bool)
    tp = int((labels & decisions).sum())
    fp = int((~labels & decisions).sum())
    fn = int((labels & ~decisions).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = 0.0 if recall is not None else None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hourly_tp": tp, "hourly_fp": fp, "hourly_fn": fn,
        "hourly_precision": precision, "hourly_recall": recall, "hourly_f1": f1,
        "hourly_alert_rate": float(decisions.mean()) if len(decisions) else None,
    }
