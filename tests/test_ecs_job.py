"""The ECS task entry point: environment variables in, /opt/ml-style folder, outputs to S3. No AWS needed."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sepsis_warning import ecs_job


class FakeS3:
    """Just the four calls the entry point makes, backed by a folder."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, bucket: str, key: str, text: str = "x") -> None:
        path = self.root / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        base = self.root / Bucket
        keys = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        keys = [k for k in keys if k.startswith(Prefix)]
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def download_file(self, bucket, key, filename):
        shutil.copy(self.root / bucket / key, filename)

    def upload_file(self, filename, bucket, key):
        self.put(bucket, key, Path(filename).read_text(encoding="utf-8"))

    def keys(self, bucket):
        base = self.root / bucket
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())


class EcsJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.s3 = FakeS3(Path(self.temp.name) / "s3")
        self.work = Path(self.temp.name) / "work"
        self.environ = {
            "HP_mode": "train", "HP_seed": "42", "HP_run_id": "exec-1", "UNRELATED": "no",
            "CH_data": "s3://bucket/data/physionet/", "CH_config": "s3://bucket/tuning/exec-1/",
            "JOB_OUTPUT": "s3://bucket/training-output/sepsis-train-exec-1/",
        }
        self.s3.put("bucket", "data/physionet/combined.parquet", "parquet")
        self.s3.put("bucket", "data/physionet/splits.csv", "csv")
        self.s3.put("bucket", "tuning/exec-1/tuned_models.toml", "toml")
        self.s3.put("bucket", "data/other/ignored.txt")

    def test_environment_becomes_hyperparameters_and_channels(self):
        self.assertEqual(ecs_job.hyperparameters(self.environ), {"mode": "train", "run_id": "exec-1", "seed": "42"})
        self.assertEqual(set(ecs_job.channels(self.environ)), {"data", "config"})

    def test_stage_builds_the_sagemaker_layout(self):
        ecs_job.stage(self.work, self.environ, self.s3)
        params = json.loads((self.work / "input" / "config" / "hyperparameters.json").read_text())
        self.assertEqual(params["mode"], "train")
        self.assertTrue((self.work / "input" / "data" / "data" / "combined.parquet").is_file())
        self.assertTrue((self.work / "input" / "data" / "data" / "splits.csv").is_file())
        self.assertTrue((self.work / "input" / "data" / "config" / "tuned_models.toml").is_file())
        self.assertFalse((self.work / "input" / "data" / "data" / "ignored.txt").exists())   # other prefix

    def test_a_successful_step_uploads_the_model_folder(self):
        def fake_main(argv=None):
            root = Path(os.environ["SM_ROOT"])
            self.assertEqual(json.loads((root / "input" / "config" / "hyperparameters.json").read_text())["seed"], "42")
            (root / "model").mkdir(parents=True, exist_ok=True)
            (root / "model" / "model.joblib").write_text("bundle")
            return 0

        with mock.patch("sepsis_warning.sagemaker_train.main", fake_main):
            code = ecs_job.run(self.environ, self.s3, workdir=self.work)
        self.assertEqual(code, 0)
        self.assertIn("training-output/sepsis-train-exec-1/model/model.joblib", self.s3.keys("bucket"))

    def test_a_failed_step_uploads_the_reason_and_returns_its_exit_code(self):
        def failing_main(argv=None):
            root = Path(os.environ["SM_ROOT"])
            (root / "output").mkdir(parents=True, exist_ok=True)
            (root / "output" / "failure").write_text("ValueError: gates not met")
            return 1

        with mock.patch("sepsis_warning.sagemaker_train.main", failing_main):
            code = ecs_job.run(self.environ, self.s3, workdir=self.work)
        self.assertEqual(code, 1)
        failure = self.s3.root / "bucket" / "training-output" / "sepsis-train-exec-1" / "output" / "failure"
        self.assertEqual(failure.read_text(), "ValueError: gates not met")

    def test_a_missing_channel_is_a_reported_failure_not_a_crash(self):
        environ = dict(self.environ, CH_data="s3://bucket/data/missing/")
        ran = []
        with mock.patch("sepsis_warning.sagemaker_train.main", lambda argv=None: ran.append(1) or 0):
            # an empty channel is staged as an empty folder; the entry point's own check then fails the step
            code = ecs_job.run(environ, self.s3, workdir=self.work)
        self.assertEqual(code, 0)
        self.assertEqual(ran, [1])
        broken = dict(self.environ, CH_data="not-an-s3-uri")
        code = ecs_job.run(broken, self.s3, workdir=Path(self.temp.name) / "work2")
        self.assertEqual(code, 1)
        self.assertIn("not an s3:// URI", (self.s3.root / "bucket" / "training-output" / "sepsis-train-exec-1" /
                                           "output" / "failure").read_text())

    def test_object_keys_cannot_escape_the_channel_folder(self):
        self.s3.put("bucket", "data/physionet/../escape.txt", "x")
        # a hostile key written straight into the listing
        original = self.s3.list_objects_v2
        self.s3.list_objects_v2 = lambda **kw: {"Contents": [{"Key": "data/physionet/../../evil"}], "IsTruncated": False}
        with self.assertRaisesRegex(ValueError, "escapes"):
            ecs_job.download_prefix("s3://bucket/data/physionet/", self.work / "ch", self.s3)
        self.s3.list_objects_v2 = original

    def test_job_output_is_required(self):
        self.assertEqual(ecs_job.run({"HP_mode": "train"}, self.s3, workdir=self.work), 2)


if __name__ == "__main__":
    unittest.main()
