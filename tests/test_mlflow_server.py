"""The MLflow server launcher for ECS: URI building and command line, without starting a server."""

import os
import unittest
from unittest import mock

from sepsis_warning import mlflow_server

ENV = {"DB_HOST": "db.example.internal", "DB_NAME": "mlflow", "DB_USER": "mlflow",
       "DB_PASSWORD": "p@ss/word:with?odd#chars", "ARTIFACTS_DESTINATION": "s3://bucket/mlflow-artifacts"}


class MlflowServerTests(unittest.TestCase):
    def test_the_password_is_percent_encoded_in_the_uri(self):
        uri = mlflow_server.backend_uri(ENV)
        self.assertEqual(uri, "postgresql://mlflow:p%40ss%2Fword%3Awith%3Fodd%23chars@db.example.internal:5432/mlflow?sslmode=require")

    def test_the_command_proxies_artifacts_to_s3_and_never_contains_the_password(self):
        cmd = mlflow_server.command(ENV)
        self.assertIn("--serve-artifacts", cmd)
        self.assertEqual(cmd[cmd.index("--artifacts-destination") + 1], "s3://bucket/mlflow-artifacts")
        self.assertEqual(cmd[cmd.index("--allowed-hosts") + 1], "*")
        self.assertFalse(any("p@ss" in part or "p%40ss" in part for part in cmd))

    def test_missing_settings_stop_the_task_with_a_reason(self):
        for name in ("DB_HOST", "DB_PASSWORD"):
            env = {k: v for k, v in ENV.items() if k != name}
            with self.assertRaisesRegex(ValueError, name):
                mlflow_server.backend_uri(env)
        with self.assertRaisesRegex(ValueError, "s3://"):
            mlflow_server.command(dict(ENV, ARTIFACTS_DESTINATION="/tmp/x"))
        self.assertEqual(mlflow_server.main({}, execvp=lambda *a: None), 2)

    def test_the_server_is_started_with_the_uri_in_the_environment_only(self):
        calls = []
        with mock.patch.dict(os.environ, {}, clear=False):
            mlflow_server.main(dict(ENV), execvp=lambda file, args: calls.append((file, args)))
            self.assertIn("p%40ss", os.environ["MLFLOW_BACKEND_STORE_URI"])
            os.environ.pop("MLFLOW_BACKEND_STORE_URI", None)
        self.assertEqual(calls[0][0], "mlflow")
        self.assertEqual(calls[0][1][:2], ["mlflow", "server"])


if __name__ == "__main__":
    unittest.main()
