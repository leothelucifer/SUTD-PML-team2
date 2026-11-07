"""E04: the saved split is reproducible, complete, and independent of load order."""

import json
import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sepsis_warning.data import generate_synthetic, load_patients, split_patients
from sepsis_warning.prepare import prepare_parquet
from sepsis_warning.splits import load_splits, write_splits


def assignment(splits):
    return {p.patient_id: name for name, cohort in splits.items() for p in cohort}


class Splits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        generate_synthetic(root / "raw", source_size=120, external_size=30)
        cls.patients = load_patients(root / "raw")
        cls.parquet = root / "combined.parquet"
        prepare_parquet(root / "raw", cls.parquet)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_load_order_does_not_change_the_split(self):
        shuffled = self.patients[:]
        random.Random(1).shuffle(shuffled)
        self.assertEqual(assignment(split_patients(self.patients, 42)), assignment(split_patients(shuffled, 42)))

    def test_saved_split_matches_psv_split_and_is_reproducible(self):
        first, second = self.root / "s1.csv", self.root / "s2.csv"
        summary = write_splits(self.parquet, first, 42)
        write_splits(self.parquet, second, 42)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        # Same assignment as splitting the .psv-loaded patients directly.
        self.assertEqual(load_splits(first), assignment(split_patients(self.patients, 42)))
        saved = json.loads((self.root / "s1.csv.summary.json").read_text())
        self.assertEqual(saved["seed"], 42)
        self.assertEqual(saved["splits_csv_sha256"], summary["splits_csv_sha256"])

    def test_every_stay_once_and_b_is_external(self):
        path = self.root / "s3.csv"
        write_splits(self.parquet, path, 42)
        table = pd.read_csv(path, dtype=str)
        self.assertEqual(len(table), len(self.patients))
        self.assertFalse(table.duplicated(["hospital", "patient_id"]).any())
        self.assertTrue((table.loc[table.hospital == "B", "split"] == "external_test").all())
        self.assertFalse((table.loc[table.hospital == "A", "split"] == "external_test").any())


if __name__ == "__main__":
    unittest.main()
