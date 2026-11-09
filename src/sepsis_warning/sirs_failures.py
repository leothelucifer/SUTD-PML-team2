"""Where the SIRS rule fails: aggregate patterns on the validation split only.

The tests and hospital B stay out of this, because looking for patterns is a
form of tuning. Every output is a group total, never a patient row, and any
group with fewer than ``MIN_CELL`` stays has its numbers blanked.
"""

import numpy as np
import pandas as pd

from .baselines import BASELINE_THRESHOLD, SIRS_ALERT_COUNT, SIRS_COLUMNS, SIRS_CRITERIA, baseline_table, sirs_criteria
from .data import Patient
from .hospital_comparison import AGE_BINS, AGE_LABELS
from .policy import apply_policy

MIN_CELL = 10
HOUR_BINS = [0, 6, 24, 48, np.inf]
HOUR_LABELS = ["1-6", "7-24", "25-48", "49+"]
NAMES = list(SIRS_CRITERIA)
# Measurements behind each criterion; "never measured" means all of them are blank so far.
SOURCES = {name: [column for column, _ in tests] for name, tests in SIRS_CRITERIA.items()}


def hourly_table(patients: list[Patient]) -> pd.DataFrame:
    """One row per stay-hour with each criterion, its measurement status and the outcome."""
    chunks = []
    for patient in patients:
        frame = patient.frame
        met = sirs_criteria(frame)
        seen = frame[SIRS_COLUMNS].notna().cummax()
        for name, columns in SOURCES.items():
            met[f"never_{name}"] = ~seen[columns].any(axis=1)
        met["stay"] = patient.patient_id
        met["hour"] = frame["ICULOS"].to_numpy()
        met["label"] = frame["SepsisLabel"].to_numpy(dtype=int)
        met["age"] = frame["Age"].to_numpy()
        met["gender"] = frame["Gender"].to_numpy()
        chunks.append(met)
    table = pd.concat(chunks, ignore_index=True)
    table["count"] = table[NAMES].sum(axis=1)
    table["flagged"] = table["count"] >= SIRS_ALERT_COUNT
    return table


def false_alarm_causes(hours: pd.DataFrame) -> pd.DataFrame:
    """Which criteria were met in the flagged hours that carry no sepsis label."""
    false_alarms = hours[hours["flagged"] & (hours["label"] == 0)]
    names = false_alarms[NAMES].apply(lambda row: " + ".join(n.removeprefix("sirs_") for n in NAMES if row[n]), axis=1)
    counts = names.value_counts().rename_axis("criteria_met").reset_index(name="false_alarm_hours")
    counts["share"] = counts["false_alarm_hours"] / len(false_alarms)
    return counts


def caught_vs_missed(hours: pd.DataFrame) -> pd.DataFrame:
    """For septic-labelled hours: how often each criterion was met or never measured."""
    positive = hours[hours["label"] == 1]
    caught, missed = positive[positive["flagged"]], positive[~positive["flagged"]]
    rows = [{
        "criterion": name.removeprefix("sirs_"),
        "met_in_caught_hours": caught[name].mean(),
        "met_in_missed_hours": missed[name].mean(),
        "never_measured_in_caught_hours": caught[f"never_{name}"].mean(),
        "never_measured_in_missed_hours": missed[f"never_{name}"].mean(),
    } for name in NAMES]
    return pd.DataFrame(rows).assign(caught_hours=len(caught), missed_hours=len(missed))


def _rates(group: pd.DataFrame) -> dict:
    tp = int((group["flagged"] & (group["label"] == 1)).sum())
    fp = int((group["flagged"] & (group["label"] == 0)).sum())
    fn = int((~group["flagged"] & (group["label"] == 1)).sum())
    tn = int((~group["flagged"] & (group["label"] == 0)).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "alert_rate": (tp + fp) / len(group) if len(group) else None,
    }


def by_hour_of_stay(hours: pd.DataFrame) -> pd.DataFrame:
    """Performance by how long the patient has been in the ICU."""
    bands = pd.cut(hours["hour"], HOUR_BINS, labels=HOUR_LABELS)
    rows = []
    for label in HOUR_LABELS:
        group = hours[bands == label]
        rows.append({"icu_hour": label, "stays": group["stay"].nunique(), "hours": len(group),
                     "positive_hours": int(group["label"].sum()), **_rates(group)})
    return _hide_small(pd.DataFrame(rows))


def by_group(hours: pd.DataFrame) -> pd.DataFrame:
    """Performance by age band and recorded gender (blank age or gender is left out)."""
    groups = {}
    ages = pd.cut(hours["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    for label in AGE_LABELS:
        groups[("age", label)] = hours[ages == label]
    for code, label in [(0, "female"), (1, "male")]:
        groups[("gender", label)] = hours[hours["gender"] == code]
    rows = []
    for (kind, label), group in groups.items():
        septic_stays = group.loc[group["label"] == 1, "stay"].nunique()
        rows.append({"group": kind, "value": label, "stays": group["stay"].nunique(),
                     "septic_stays": septic_stays, "hours": len(group), **_rates(group)})
    return _hide_small(pd.DataFrame(rows), "septic_stays")


def missed_stays(patients: list[Patient], cooldown: int = 6) -> pd.DataFrame:
    """Septic stays warned in the 12 h before onset vs not warned, same rule as early detection."""
    scored = apply_policy(baseline_table(patients, "sirs"), BASELINE_THRESHOLD, cooldown)
    frames = {p.patient_id: p.frame for p in patients}
    records = []
    for stay, patient in scored.groupby("patient_id", sort=False):
        patient = patient.sort_values("hour")
        positive = np.flatnonzero(patient["label"].to_numpy())
        if not len(positive) or positive[0] == 0:
            continue  # no sepsis, or onset unknown because positive from the first row
        onset = float(patient.iloc[positive[0]]["hour"]) + 6
        if onset > float(patient["hour"].max()):
            continue
        window = patient[(patient["hour"] >= onset - 12) & (patient["hour"] < onset)]
        frame = frames[stay]
        before = frame[frame["ICULOS"] < onset]
        peak = (window["probability"].max() * len(SIRS_CRITERIA)) if len(window) else 0
        records.append({
            "warned": bool(window["alert_event"].sum()),
            "hours_before_onset": len(before),
            "wbc_seen_before_onset": bool(before["WBC"].notna().any()),
            "peak_criteria_in_window": float(peak),
        })
    records = pd.DataFrame(records, columns=["warned", "hours_before_onset", "wbc_seen_before_onset", "peak_criteria_in_window"])
    rows = []
    for label, group in [("warned", records[records["warned"]]), ("not warned", records[~records["warned"].astype(bool)])]:
        rows.append({
            "outcome": label, "stays": len(group),
            "median_hours_before_onset": group["hours_before_onset"].median(),
            "share_with_wbc_before_onset": group["wbc_seen_before_onset"].mean(),
            "median_peak_criteria_in_window": group["peak_criteria_in_window"].median(),
        })
    return _hide_small(pd.DataFrame(rows))


def _hide_small(table: pd.DataFrame, column: str = "stays") -> pd.DataFrame:
    """Blank every number in a row built from fewer than MIN_CELL stays."""
    table = table.astype({c: "float64" for c in table.select_dtypes("number").columns})
    small = table[column] < MIN_CELL
    table.loc[small, table.select_dtypes("number").columns] = np.nan
    return table


def failure_tables(patients: list[Patient], cooldown: int = 6) -> dict[str, pd.DataFrame]:
    hours = hourly_table(patients)
    return {
        "false_alarm_causes": false_alarm_causes(hours),
        "caught_vs_missed_hours": caught_vs_missed(hours),
        "by_hour_of_stay": by_hour_of_stay(hours),
        "by_group": by_group(hours),
        "warned_vs_not_warned_stays": missed_stays(patients, cooldown),
    }
