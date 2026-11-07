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


def sirs_threshold_sweep(table: pd.DataFrame, cooldown: int = 6) -> pd.DataFrame:
    """Course L4 threshold trade-off for SIRS: alert at >= k criteria, k = 1..4.

    Run this on the VALIDATION split only. It is descriptive: the official rule
    stays at k = 2 and is not re-chosen from these numbers.
    ``table`` is a ``baseline_table(..., "sirs")`` result (probability = count/4).
    """
    from .policy import apply_policy, summarize  # local import avoids a cycle

    rows = []
    for k in range(1, len(SIRS_CRITERIA) + 1):
        scored = apply_policy(table, k / len(SIRS_CRITERIA), cooldown)
        metrics = summarize(scored)
        metrics.update(hourly_classification_metrics(scored["label"], scored["hourly_positive"]))
        rows.append({"alert_at_criteria": k, **metrics})
    return pd.DataFrame(rows)


def plot_sirs_threshold_tradeoff(sweep: pd.DataFrame, positive_rate: float, budget: float, path, recreate: str):
    """Left: precision-recall points for >= 1..4 criteria. Right: false alerts per day."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    left.plot(sweep["hourly_recall"], sweep["hourly_precision"], "-o", color="#2a6fdb")
    for _, row in sweep.iterrows():
        k = int(row["alert_at_criteria"])
        left.annotate(f"≥{k}" + (" (rule)" if k == SIRS_ALERT_COUNT else ""),
                      (row["hourly_recall"], row["hourly_precision"]), textcoords="offset points",
                      xytext=(6, 4), fontsize=9, fontweight="bold" if k == SIRS_ALERT_COUNT else "normal")
    left.axhline(positive_rate, color="grey", ls="--", lw=1)
    left.text(0.02, positive_rate, f" random guessing = positive rate {positive_rate:.3f}", va="bottom",
              fontsize=8, color="grey", transform=left.get_yaxis_transform())
    left.set_xlim(0, 1.02)
    left.set_ylim(0, max(sweep["hourly_precision"].max() * 1.6, positive_rate * 3))
    left.set_xlabel("Recall (septic-labelled hours flagged)")
    left.set_ylabel("Precision (flagged hours that are septic)")
    left.set_title("Precision vs recall")
    rate = sweep["false_alert_events_per_nonseptic_patient_day"]
    right.bar(sweep["alert_at_criteria"].astype(str).radd("≥"), rate,
              color=["#2a6fdb" if k == SIRS_ALERT_COUNT else "#9dbcf0" for k in sweep["alert_at_criteria"]])
    right.axhline(budget, color="#d96f2c", lw=1.5)
    right.text(-0.45, budget, f"budget {budget}/day", va="bottom", ha="left", fontsize=8, color="#d96f2c")
    for x, value in enumerate(rate):
        right.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    right.set_xlabel("Alert when this many SIRS criteria are met")
    right.set_ylabel("False alerts per non-septic patient-day")
    right.set_title("Alarm burden (6 h cooldown)")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("8. SIRS threshold trade-off (validation split only)")
    fig.text(0.99, 0.005, f"Recreate: {recreate}", ha="right", va="bottom", fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hourly_classification_metrics(labels, decisions) -> dict:
    """Hand-written hourly confusion-matrix metrics (course L9; no library metrics).

    precision   = TP / (TP + FP)      share of flagged hours that are septic-labelled
    recall      = TP / (TP + FN)      share of septic-labelled hours that are flagged
    specificity = TN / (TN + FP)      share of non-septic hours left alone
    F1          = 2TP / (2TP + FP + FN)
    accuracy    = (TP + TN) / all     reported because the brief asks for it, but
                                      misleading here: always-no already gets ~98%.

    When nothing is flagged (e.g. always-no), precision is 0/0. Following the
    course convention it is reported as 0 and flagged in ``no_predicted_positives``.
    """
    labels = np.asarray(labels, dtype=bool)
    decisions = np.asarray(decisions, dtype=bool)
    tp = int((labels & decisions).sum())
    fp = int((~labels & decisions).sum())
    fn = int((labels & ~decisions).sum())
    tn = int((~labels & ~decisions).sum())
    total = tp + fp + fn + tn
    return {
        "hourly_tp": tp, "hourly_fp": fp, "hourly_fn": fn, "hourly_tn": tn,
        "hourly_accuracy": (tp + tn) / total if total else None,
        "hourly_precision": tp / (tp + fp) if tp + fp else 0.0,
        "hourly_recall": tp / (tp + fn) if tp + fn else None,
        "hourly_specificity": tn / (tn + fp) if tn + fp else None,
        "hourly_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "hourly_alert_rate": (tp + fp) / total if total else None,
        "no_predicted_positives": tp + fp == 0,
    }
