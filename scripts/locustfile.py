"""IAM-SigV4 Locust workload for POST /predict."""

import json
import os
import random
import sys
import uuid
from pathlib import Path

import boto3
from locust import HttpUser, between, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
from call_scoring_url import signed_request


def load_payloads(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        payloads = [json.loads(line) for line in source if line.strip()]
    if not payloads:
        raise ValueError(f"No Locust payloads found in {path}")
    return payloads


class SigV4PredictionUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        payload_path = os.environ.get("LOCUST_PAYLOADS")
        if not payload_path:
            raise RuntimeError("LOCUST_PAYLOADS must point to a private Hospital B JSONL file")
        self.payloads = load_payloads(Path(payload_path))
        self.region = os.environ.get("AWS_REGION") or boto3.Session().region_name
        if not self.region:
            raise RuntimeError("AWS_REGION is required to sign Lambda Function URL requests")
        self.credentials = boto3.Session().get_credentials()
        if self.credentials is None:
            raise RuntimeError("No AWS credentials available for SigV4 signing")

    @task
    def predict(self):
        stay = dict(random.choice(self.payloads))
        stay["patient_id"] = f"locust-{uuid.uuid4().hex}"
        body = json.dumps({"stays": [stay]}, allow_nan=False, separators=(",", ":"))
        url = f"{self.host.rstrip('/')}/predict"
        request = signed_request(url, "POST", body, self.region, self.credentials)
        with self.client.post(
            "/predict",
            data=request.body,
            headers=dict(request.headers),
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                result = response.json()
            except ValueError:
                response.failure("invalid JSON response")
                return
            if not isinstance(result, dict) or not isinstance(result.get("predictions"), list) or not result["predictions"]:
                response.failure("invalid prediction response")
                return
            response.success()
