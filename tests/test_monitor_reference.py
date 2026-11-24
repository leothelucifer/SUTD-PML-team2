"""The monitor_reference pipeline mode: the Evidently reference comes from validation only and holds no identifiers."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pandas as pd

from sepsis_warning import evidently_frames as frames
from sepsis_warning import sagemaker_train
from sepsis_warning.cli import run_experiment
from sepsis_warning.data import generate_synthetic
from sepsis_warning.prepare import prepare_parquet
from sepsis_warning.scoring import file_sha256
from sepsis_warning.splits import write_splits


class FakeS3:
    def __init__(self):
        self.uploaded = {}

    def upload_file(self, filename, bucket, key):
        self.uploaded[f"{bucket}/{key}"] = Path(filename).read_bytes()


class MonitorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        raw = root / "raw"
        generate_synthetic(raw, source_size=160, external_size=20)
        (raw / "SYNTHETIC.json").write_text("{}")
        cls.parquet, cls.splits = root / "combined.parquet", root / "splits.csv"
        prepare_parquet(raw, cls.parquet)
        write_splits(cls.parquet, cls.splits, 42)
        run_experiment(Namespace(data=cls.parquet, splits=cls.splits, output=root / "run", seed=42, window=6, budget=0.2,
                                 cooldown=6, splits_summary=None, dvc_lock=None, mlflow_uri=None, register_model=False))
        cls.model = root / "run" / "model.joblib"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_mode(self, **overrides):
        s3, root = FakeS3(), Path(tempfile.mkdtemp(prefix="ref-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        params = {"mlflow_tracking_uri": "http://mlflow.x.local:5000", "publish_bucket": "bkt", **overrides}
        args = Namespace(data=self.parquet, splits=self.splits)
        with mock.patch("sepsis_warning.registry.resolve_model", return_value=(self.model, file_sha256(self.model), {})):
            uris = sagemaker_train.monitor_reference(root, params, args, client=s3)
        return uris, s3

    def test_the_reference_is_stored_under_the_bundle_and_has_only_the_live_columns(self):
        uris, s3 = self.run_mode()
        sha = file_sha256(self.model)[:12]
        self.assertEqual(sorted(u.rsplit("/", 1)[-1] for u in uris), ["reference.json", "reference.parquet"])
        self.assertTrue(all(f"monitoring/evidently/{sha}/" in u for u in uris))
        key = f"bkt/monitoring/evidently/{sha}/reference.parquet"
        path = Path(tempfile.mkdtemp()) / "r.parquet"
        path.write_bytes(s3.uploaded[key])
        reference = pd.read_parquet(path)
        self.assertEqual(list(reference.columns), frames.COLUMNS)           # no id, no hour, no measurement values
        self.assertGreater(len(reference), 100)
        meta = json.loads(s3.uploaded[f"bkt/monitoring/evidently/{sha}/reference.json"])
        self.assertEqual((meta["bundle_sha256"], meta["split"]), (file_sha256(self.model), "validation"))
        self.assertEqual(meta["rows"], len(reference))

    def test_it_needs_a_registry_a_bucket_and_a_saved_split(self):
        with self.assertRaisesRegex(ValueError, "needs mlflow_tracking_uri and publish_bucket"):
            sagemaker_train.monitor_reference(Path(tempfile.mkdtemp()), {}, Namespace(data=self.parquet, splits=self.splits))
        with self.assertRaisesRegex(ValueError, "splits.csv"):
            sagemaker_train.monitor_reference(Path(tempfile.mkdtemp()), {"mlflow_tracking_uri": "x", "publish_bucket": "b"},
                                              Namespace(data=self.parquet, splits=None))


if __name__ == "__main__":
    unittest.main()
