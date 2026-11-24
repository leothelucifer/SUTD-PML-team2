"""MLflow on ECS: which tracking servers the code accepts, who is acting, and the promote step."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sepsis_warning import registry, sagemaker_train, tracking


class ProjectServerTests(unittest.TestCase):
    def test_the_projects_own_server_is_accepted_and_other_hosts_are_not(self):
        for uri in ("http://mlflow.sepsis-dev.local:5000", "http://10.50.1.20:5000", "https://mlflow.sepsis-dev.local"):
            self.assertTrue(tracking.is_project_server(uri), uri)
            self.assertEqual(tracking.check_tracking_uri(uri), uri)
        for uri in ("http://mlflow.example.com:5000", "http://54.1.2.3:5000", "ftp://x.local", "http://localhost:5000"):
            self.assertFalse(tracking.is_project_server(uri), uri)
        for uri in ("http://mlflow.example.com:5000", "http://54.1.2.3:5000"):
            with self.assertRaisesRegex(ValueError, "needs team approval"):
                tracking.check_tracking_uri(uri)
        self.assertEqual(tracking.check_tracking_uri("http://localhost:5000"), "http://localhost:5000")

    def test_a_person_can_name_the_address_they_were_given(self):
        with mock.patch.dict(os.environ, {"SEPSIS_MLFLOW_HOSTS": "54.1.2.3, other.example.org"}):
            self.assertTrue(tracking.is_project_server("http://54.1.2.3:5000"))
            self.assertTrue(tracking.is_project_server("http://other.example.org:5000"))
            self.assertFalse(tracking.is_project_server("http://54.9.9.9:5000"))

    @unittest.skipUnless(importlib.util.find_spec("boto3"), "needs `pip install -e .[aws]`")
    def test_on_aws_the_actor_is_the_task_role_not_the_local_user(self):
        with mock.patch("boto3.client") as client:
            client.return_value.get_caller_identity.return_value = {"Arn": "arn:aws:sts::1:assumed-role/sepsis-approver/task"}
            self.assertEqual(registry.caller_identity("http://mlflow.sepsis-dev.local:5000"),
                             "arn:aws:sts::1:assumed-role/sepsis-approver/task")
        self.assertTrue(registry.caller_identity("sqlite:///x.db").startswith("local:"))


class PromoteModeTests(unittest.TestCase):
    def test_promote_needs_the_registry_the_version_and_a_named_reviewer(self):
        with tempfile.TemporaryDirectory() as temp:
            for params in ({}, {"mlflow_tracking_uri": "http://mlflow.x.local:5000", "model_version": "3"},
                           {"mlflow_tracking_uri": "http://mlflow.x.local:5000", "reviewed_by": "A"}):
                with self.assertRaisesRegex(ValueError, "needs mlflow_tracking_uri"):
                    sagemaker_train.promote_model(Path(temp), params)

    def test_promote_calls_the_registry_with_the_gates_and_records_the_result(self):
        change = {"name": "sepsis-warning", "alias": "production", "version": "3", "previous": None,
                  "reviewed_by": "Reviewer", "approved_by": "arn:aws:sts::1:assumed-role/approver/x", "promoted_at": "t", "gates": []}
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch("sepsis_warning.registry.promote", return_value=change) as promote:
            sagemaker_train.promote_model(Path(temp), {"mlflow_tracking_uri": "http://mlflow.x.local:5000",
                                                       "model_version": "3", "reviewed_by": "Reviewer",
                                                       "require_real_data": "true"})
            promote.assert_called_once_with("http://mlflow.x.local:5000", "3", "Reviewer", name="sepsis-warning",
                                            alias="production", require_real_data=True)
            self.assertEqual(json.loads((Path(temp) / "model" / "promotion.json").read_text())["version"], "3")

    def test_main_runs_promote_without_a_data_channel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input" / "config").mkdir(parents=True)
            (root / "input" / "config" / "hyperparameters.json").write_text(json.dumps({"mode": "promote"}))
            with mock.patch.dict(os.environ, {"SM_ROOT": temp}):
                self.assertEqual(sagemaker_train.main([]), 1)    # it fails on the missing settings, not on the data channel
            self.assertIn("mode promote needs", (root / "output" / "failure").read_text())


if __name__ == "__main__":
    unittest.main()
