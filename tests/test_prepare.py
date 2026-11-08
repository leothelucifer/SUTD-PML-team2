"""E01: prepare keeps the data exactly, rejects corrupt records, never overwrites."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.data import DYNAMIC, STATIC
from sepsis_warning.prepare import prepare_parquet


def good_record(hours=4):
    frame = pd.DataFrame(np.nan, index=range(hours), columns=DYNAMIC + STATIC + ["Unit1", "Unit2", "SepsisLabel"])
    frame["HR"] = [80.0, np.nan, 95.5, 101.0][:hours]
    frame["Temp"] = [np.nan, 37.2, np.nan, 38.4][:hours]
    frame["Age"], frame["Gender"], frame["HospAdmTime"] = 67.3, 1, -12.5
    frame["ICULOS"] = np.arange(1, hours + 1)
    frame["SepsisLabel"] = [0, 0, 1, 1][:hours]
    return frame


def write_cohort(root, a_record=None, b_record=None):
    for hospital, frame in [("A", a_record), ("B", b_record)]:
        directory = root / f"training_set{hospital}"
        directory.mkdir(parents=True, exist_ok=True)
        frame = good_record() if frame is None else frame
        if isinstance(frame, str):
            (directory / "p000001.psv").write_text(frame)
        else:
            frame.to_csv(directory / "p000001.psv", sep="|", index=False, na_rep="NaN")


class Prepare(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output = self.root / "out" / "combined.parquet"

    def tearDown(self):
        self._tmp.cleanup()

    def test_values_and_missingness_are_preserved_exactly(self):
        write_cohort(self.root)
        prepare_parquet(self.root, self.output)
        combined = pd.read_parquet(self.output)
        original = pd.read_csv(self.root / "training_setA" / "p000001.psv", sep="|")
        stored = combined[combined.hospital == "A"].drop(columns=["patient_id", "hospital"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(stored, original.astype("float64"))
        self.assertEqual(int(stored["HR"].isna().sum()), 1)  # NaN stays NaN, never filled
        self.assertEqual(set(combined["patient_id"]), {"p000001"})
        self.assertEqual(sorted(combined["hospital"].unique()), ["A", "B"])

    def test_refuses_to_overwrite_existing_output(self):
        write_cohort(self.root)
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            prepare_parquet(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep me")

    def test_cli_reports_existing_output_without_traceback(self):
        write_cohort(self.root)
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"keep me")
        result = subprocess.run(
            [sys.executable, "-m", "sepsis_warning", "prepare", "--data", str(self.root), "--output", str(self.output)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_corrupt_records_are_rejected_and_nothing_is_written(self):
        def broken(change):
            frame = good_record()
            change(frame)
            return frame

        # name: (record, part of the error message that proves the right check fired)
        cases = {
            "label not 0/1": (broken(lambda f: f.__setitem__("SepsisLabel", [0, 0, 2, 2])), "binary, nondecreasing"),
            "label goes back to 0": (broken(lambda f: f.__setitem__("SepsisLabel", [0, 1, 0, 0])), "binary, nondecreasing"),
            "skipped hour": (broken(lambda f: f.__setitem__("ICULOS", [1, 2, 4, 5])), "consecutive hourly"),
            "missing column": (good_record().drop(columns=["WBC"]), "missing columns"),
            "infinite value": (broken(lambda f: f.loc.__setitem__((1, "HR"), np.inf)), "infinite"),
            "text in a number column": (broken(lambda f: f.__setitem__("HR", ["abc", 1, 2, 3])), "Unable to parse"),
            "empty record": ("|".join(DYNAMIC + STATIC + ["Unit1", "Unit2", "SepsisLabel"]) + "\n", "empty record"),
        }
        for name, (record, message) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_cohort(root, b_record=record)
                output = root / "combined.parquet"
                with self.assertRaisesRegex(ValueError, message):
                    prepare_parquet(root, output)
                self.assertFalse(output.exists())
                self.assertFalse(any(root.glob(".*.tmp")), "temporary file left behind")

    def test_schema_must_match_across_records(self):
        extra = good_record().assign(Extra=1.0)
        write_cohort(self.root, b_record=extra)
        with self.assertRaisesRegex(ValueError, "schema differs"):
            prepare_parquet(self.root, self.output)


if __name__ == "__main__":
    unittest.main()
