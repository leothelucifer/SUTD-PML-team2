"""SIRS / always-no baseline: rule correctness, causality, missing handling."""

import unittest

import numpy as np
import pandas as pd

from sepsis_warning.baselines import (
    BASELINE_THRESHOLD, baseline_table, hourly_classification_metrics, sirs_count,
)
from sepsis_warning.data import DYNAMIC, STATIC, Patient
from sepsis_warning.policy import apply_policy


def make_patient(hours, label_from=None, **values):
    """All-missing patient for `hours` hours; overwrite chosen columns."""
    frame = pd.DataFrame(np.nan, index=range(hours), columns=DYNAMIC + STATIC)
    frame["ICULOS"] = np.arange(1, hours + 1)
    frame["SepsisLabel"] = 0
    if label_from is not None:
        frame.loc[label_from:, "SepsisLabel"] = 1
    for column, series in values.items():
        frame[column] = series
    return frame


class SirsBaseline(unittest.TestCase):
    def test_fires_at_exact_expected_hour(self):
        nan = np.nan
        frame = make_patient(
            6,
            HR=[80, 95, nan, nan, nan, nan],        # abnormal from row 1 on (ffill)
            Temp=[37, nan, nan, 38.5, nan, nan],    # abnormal from row 3 on
            Resp=[18, nan, nan, nan, nan, 22],      # abnormal at row 5
        )
        np.testing.assert_array_equal(sirs_count(frame), [0, 1, 1, 2, 2, 3])
        table = baseline_table([Patient("A:p1", "A", frame)], "sirs")
        alerts = apply_policy(table, BASELINE_THRESHOLD, cooldown=6)
        np.testing.assert_array_equal(alerts["hourly_positive"], [0, 0, 0, 1, 1, 1])
        # First alert at row 3; cooldown suppresses rows 4 and 5.
        np.testing.assert_array_equal(alerts["alert_event"], [0, 0, 0, 1, 0, 0])

    def test_or_criteria_and_thresholds_are_strict(self):
        frame = make_patient(
            5,
            HR=[90, 90, 91, 90, 90],                # boundary 90 is NOT > 90
            Temp=[38, 36, 37, 35.9, 37],            # 38 and 36 are normal
            PaCO2=[40, 40, 40, 40, 31],             # PaCO2 < 32 meets resp criterion
            WBC=[12, 4, 8, 8, 3.9],                 # 12 and 4 are normal
        )
        np.testing.assert_array_equal(sirs_count(frame), [0, 0, 1, 1, 2])

    def test_future_hours_never_change_earlier_scores(self):
        rng = np.random.default_rng(0)
        frame = make_patient(
            30, HR=rng.normal(90, 10, 30), Temp=rng.normal(37.5, 1, 30),
            Resp=rng.normal(20, 4, 30), WBC=rng.normal(10, 4, 30),
        )
        frame.loc[rng.random(30) < 0.5, "Temp"] = np.nan
        before = sirs_count(frame)[:15]
        changed = frame.copy()
        changed.loc[15:, ["HR", "Temp", "Resp", "PaCO2", "WBC"]] = [200, 42, 40, 10, 30]
        changed.loc[15:, "SepsisLabel"] = 1
        np.testing.assert_array_equal(before, sirs_count(changed)[:15])
        for length in [1, 7, 15]:
            np.testing.assert_array_equal(before[:length], sirs_count(frame.iloc[:length]))

    def test_missing_values_never_count_as_abnormal(self):
        frame = make_patient(4)                     # nothing ever observed
        np.testing.assert_array_equal(sirs_count(frame), [0, 0, 0, 0])
        # A value observed later is not back-filled into earlier hours.
        frame = make_patient(4, HR=[np.nan, np.nan, 120, np.nan], Temp=[np.nan, np.nan, 39, np.nan])
        np.testing.assert_array_equal(sirs_count(frame), [0, 0, 2, 2])

    def test_always_no_never_alerts(self):
        frame = make_patient(10, label_from=4, HR=[120] * 10, Temp=[39] * 10)
        table = apply_policy(baseline_table([Patient("A:p", "A", frame)], "always_no"), BASELINE_THRESHOLD, 6)
        self.assertEqual(table["alert_event"].sum(), 0)
        metrics = hourly_classification_metrics(table["label"], table["hourly_positive"])
        self.assertIsNone(metrics["hourly_precision"])
        self.assertEqual(metrics["hourly_recall"], 0)
        self.assertEqual(metrics["hourly_f1"], 0)

    def test_hand_written_metrics(self):
        metrics = hourly_classification_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
        self.assertAlmostEqual(metrics["hourly_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["hourly_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["hourly_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
