"""The MLflow server launcher for ECS: URI building and command line, without starting a server."""

import os
import unittest
from unittest import mock

from sepsis_warning.lifecycle import mlflow_server

ENV = {"DB_HOST": "db.example.internal", "DB_NAME": "mlflow", "DB_USER": "mlflow",
       "DB_PASSWORD": "p@ss/word:with?odd#chars", "ARTIFACTS_DESTINATION": "s3://bucket/mlflow-artifacts"}


class MlflowServerTests(unittest.TestCase):
    def test_the_password_is_percent_encoded_in_the_uri(self):
        uri = mlflow_server.backend_uri(ENV)
        self.assertEqual(uri, "postgresql+psycopg2://mlflow:p%40ss%2Fword%3Awith%3Fodd%23chars@db.example.internal:5432/mlflow?sslmode=require")

    def test_the_database_driver_is_the_one_the_image_installs(self):
        # The first MLflow task on AWS crashed with "No module named 'psycopg'": SQLAlchemy 2.1 maps a plain postgresql://
        # URL to psycopg (version 3), and the image has psycopg2.
        from pathlib import Path

        from sqlalchemy.engine import make_url

        uri = mlflow_server.backend_uri(ENV)
        self.assertEqual(make_url(uri).get_dialect().driver, "psycopg2")
        lock = (Path(__file__).resolve().parents[1] / "requirements" / "tools.txt").read_text(encoding="utf-8")
        self.assertIn("psycopg2-binary==", lock)
        self.assertNotIn("\npsycopg==", lock)
        self.assertNotIn("\npsycopg[", lock)
        from mlflow.utils.uri import extract_db_type_from_uri

        self.assertEqual(extract_db_type_from_uri(uri), "postgresql")      # MLflow accepts the driver suffix

    def test_the_command_proxies_artifacts_to_s3_and_never_contains_the_password(self):
        cmd = mlflow_server.command(ENV)
        self.assertIn("--serve-artifacts", cmd)
        self.assertEqual(cmd[cmd.index("--artifacts-destination") + 1], "s3://bucket/mlflow-artifacts")
        self.assertEqual(cmd[cmd.index("--allowed-hosts") + 1], "*")
        self.assertEqual(cmd[cmd.index("--cors-allowed-origins") + 1], "*")
        self.assertFalse(any("p@ss" in part or "p%40ss" in part for part in cmd))

    def test_the_origins_can_be_narrowed_from_the_environment(self):
        cmd = mlflow_server.command(dict(ENV, MLFLOW_CORS_ALLOWED_ORIGINS="http://203.0.113.7:5000"))
        self.assertEqual(cmd[cmd.index("--cors-allowed-origins") + 1], "http://203.0.113.7:5000")

    def test_the_browser_ui_can_search_runs_on_the_server_this_command_starts(self):
        """The UI's search calls carry the page's origin; with only --allowed-hosts MLflow answered 403 and every run list stayed empty."""
        import json
        import socket
        import subprocess
        import tempfile
        import time
        import urllib.error
        import urllib.request

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as temp:
            cmd = mlflow_server.command(dict(ENV, PORT=str(port)))
            cmd[cmd.index("--host") + 1] = "127.0.0.1"
            env = {**os.environ, "MLFLOW_BACKEND_STORE_URI": f"sqlite:///{temp}/m.db", "MLFLOW_DISABLE_TELEMETRY": "true",
                   "MLFLOW_DISABLE_AGENT_HINT": "1"}
            server = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(60):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                        break
                    except (urllib.error.URLError, OSError):
                        time.sleep(1)
                else:
                    self.fail("the MLflow server did not start")
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/ajax-api/2.0/mlflow/runs/search", method="POST",
                    data=json.dumps({"experiment_ids": ["0"], "max_results": 5}).encode(),
                    headers={"Content-Type": "application/json", "Origin": "http://3.27.208.97:5000"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.terminate()
                server.wait(timeout=15)

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
