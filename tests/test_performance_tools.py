import csv
import json
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_locust_payloads import prepare_payloads, write_private_jsonl
from scripts.summarize_locust import aggregate_stats, build_report, render_markdown
from sepsis_warning.data.records import generate_synthetic


class PerformanceToolsTests(unittest.TestCase):
    def test_prepare_locust_payloads_anonymizes_and_excludes_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            generate_synthetic(data_root, source_size=20, external_size=8)

            payloads = prepare_payloads(data_root, limit=3, seed=7)

        self.assertEqual(len(payloads), 3)
        self.assertEqual(len({payload["patient_id"] for payload in payloads}), 3)
        self.assertTrue(all(payload["patient_id"].startswith("hospital-b-") for payload in payloads))
        self.assertTrue(all(
            "SepsisLabel" not in hour and "Unit1" not in hour and "Unit2" not in hour
            for payload in payloads for hour in payload["hours"]
        ))
        self.assertTrue(all(
            hour["ICULOS"] == index + 1
            for payload in payloads for index, hour in enumerate(payload["hours"])
        ))

    def test_private_payloads_are_strict_json_and_owner_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "payloads.jsonl"
            write_private_jsonl(output, [{"patient_id": "hospital-b-0000", "hours": [{"ICULOS": 1, "HR": None}]}])

            self.assertEqual(json.loads(output.read_text())["patient_id"], "hospital-b-0000")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_private_payloads_cannot_be_written_into_the_repository(self):
        output = Path(__file__).resolve().parents[1] / "accidental-patient-payloads.jsonl"

        with self.assertRaisesRegex(ValueError, "outside the repository"):
            write_private_jsonl(output, [])

    def test_locust_summary_contains_only_aggregate_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            stats = tmp_path / "stats.csv"
            with stats.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=[
                    "Type", "Name", "Request Count", "Failure Count", "Requests/s", "Average Response Time",
                    "Median Response Time", "95%", "99%", "Min Response Time", "Max Response Time", "Average Content Size",
                ])
                writer.writeheader()
                writer.writerow({
                    "Type": "", "Name": "Aggregated", "Request Count": "120", "Failure Count": "2",
                    "Requests/s": "2.0", "Average Response Time": "130", "Median Response Time": "100",
                    "95%": "250", "99%": "400", "Min Response Time": "40", "Max Response Time": "500",
                    "Average Content Size": "2048",
                })
            manifest = tmp_path / "manifest.json"
            manifest.write_text(json.dumps({
                "cohort": "hospital B", "source": "PhysioNet", "records": 3, "hours": 120, "seed": 42,
            }))
            health = tmp_path / "health.json"
            health.write_text(json.dumps({
                "synthetic_model": False, "model_name": "frozen-model", "model_version": "1",
                "bundle_sha256": "abc123",
            }))

            metrics = aggregate_stats(stats)
            report = build_report(stats, manifest, health, {
                "run_id": "123", "environment": "dev", "region": "ap-southeast-1",
                "users": 5, "spawn_rate": 1, "duration": "60s",
            })
            summary = render_markdown(report)

        self.assertEqual(metrics["requests"], 120)
        self.assertEqual(metrics["failures"], 2)
        self.assertEqual(metrics["p95_ms"], 250)
        self.assertIn("Hospital B records: 3 (120 hourly rows)", summary)
        self.assertNotIn("abc123", summary)
        self.assertIn("are temporary and are not included", summary)

    def test_summary_refuses_synthetic_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            stats = tmp_path / "stats.csv"
            stats.write_text(
                "Type,Name,Request Count,Failure Count,Requests/s,Average Response Time,Median Response Time,95%,99%,"
                "Min Response Time,Max Response Time,Average Content Size\n"
                ",Aggregated,1,0,1,10,10,10,10,10,10,1024\n"
            )
            manifest = tmp_path / "manifest.json"
            manifest.write_text(json.dumps({
                "cohort": "hospital B", "source": "PhysioNet", "records": 1, "hours": 10,
            }))
            health = tmp_path / "health.json"
            health.write_text(json.dumps({"synthetic_model": True}))

            with self.assertRaisesRegex(ValueError, "non-synthetic"):
                build_report(stats, manifest, health, {})
