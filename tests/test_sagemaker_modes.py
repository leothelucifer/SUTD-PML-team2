"""The cloud pipeline's job modes (prepare, tune, train with a report) run end to end on synthetic data."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from sepsis_warning import sagemaker_train


class FakeS3:
    """upload_file copies into a local folder laid out as <root>/<bucket>/<key>."""

    def __init__(self, root: Path):
        self.root = root
        self.uploaded = []

    def upload_file(self, filename, bucket, key):
        target = self.root / bucket / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, target)
        self.uploaded.append(f"{bucket}/{key}")


def job(root: Path, **params) -> None:
    config = root / "input" / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "hyperparameters.json").write_text(json.dumps({k: str(v) for k, v in params.items()}))


class ModeTests(unittest.TestCase):
    def test_prepare_tune_and_train_with_a_report(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            s3 = FakeS3(temp / "s3")

            # 1. prepare: synthetic data built and uploaded, with nothing on disk to start from
            prepare = temp / "prepare"
            params = dict(mode="prepare", data_source="synthetic", synthetic_stays=120, publish_bucket="bkt", seed=42)
            job(prepare, **params)
            uris = sagemaker_train.prepare_data(prepare, sagemaker_train._params(prepare), client=s3)
            self.assertEqual(sorted(Path(u).name for u in uris), ["combined.parquet", "splits.csv", "splits.csv.summary.json"])
            data_dir = temp / "s3" / "bkt" / "data" / "physionet"
            self.assertTrue((data_dir / "combined.parquet").is_file())

            # 2. tune: reads the data channel, writes tuned_models.toml under tuning/<run id>/
            tune = temp / "tune"
            shutil.copytree(data_dir, tune / "input" / "data" / "data")
            job(tune, mode="tune", trials=1, folds=2, publish_bucket="bkt", run_id="run-1", seed=42)
            args = sagemaker_train.build_args(tune)
            out = sagemaker_train.tune_data(tune, sagemaker_train._params(tune), args, client=s3)
            self.assertTrue((out / "tuned_models.toml").is_file())
            self.assertIn("bkt/tuning/run-1/tuned_models.toml", s3.uploaded)
            self.assertIn("bkt/tuning/run-1/tuning_results.csv", s3.uploaded)

            # 3. train: picks the tuned settings up from the config channel and writes the report
            train = temp / "train"
            shutil.copytree(data_dir, train / "input" / "data" / "data")
            shutil.copytree(temp / "s3" / "bkt" / "tuning" / "run-1", train / "input" / "data" / "config")
            job(train, mode="train", publish_bucket="bkt", run_id="run-1", seed=42, bootstrap=20)
            args = sagemaker_train.build_args(train)
            self.assertEqual(args.models_config, train / "input" / "data" / "config" / "tuned_models.toml")
            with mock.patch.dict(os.environ, {"SM_ROOT": str(train)}), mock.patch.object(sagemaker_train, "S3_CLIENT", s3):
                self.assertEqual(sagemaker_train.main([]), 0)
            report = train / "model" / "report"
            for name in ("metrics_table.md", "fig_pr_curves.png", "fig_tuning.png", "FIGURES.md"):
                self.assertTrue((report / name).is_file(), name)
            self.assertTrue(any(u.startswith("bkt/reports/run-1/") for u in s3.uploaded))

    def test_zero_trials_passes_the_reference_settings_on(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            s3 = FakeS3(temp / "s3")
            root = temp / "tune"
            sagemaker_train.prepare_data(root, dict(data_source="synthetic", synthetic_stays=60, publish_bucket="bkt"), client=s3)
            shutil.copytree(temp / "s3" / "bkt" / "data" / "physionet", root / "input" / "data" / "data")
            job(root, mode="tune", trials=0, publish_bucket="bkt", run_id="r")
            out = sagemaker_train.tune_data(root, sagemaker_train._params(root), sagemaker_train.build_args(root), client=s3)
            self.assertIn("logistic_regression", (out / "tuned_models.toml").read_text())
            self.assertIn("bkt/tuning/r/tuned_models.toml", s3.uploaded)

    def test_unknown_data_source_and_missing_bucket_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                sagemaker_train.prepare_data(Path(temp), {"data_source": "synthetic"})
            with self.assertRaises(ValueError):
                sagemaker_train.prepare_data(Path(temp), {"data_source": "other", "publish_bucket": "b"}, client=FakeS3(Path(temp)))

    def test_a_failing_report_does_not_fail_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = mock.Mock(output=root / "missing", seed=1)
            self.assertEqual(sagemaker_train.write_run_report(root, {}, args), [])


if __name__ == "__main__":
    unittest.main()
