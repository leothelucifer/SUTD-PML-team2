"""The first request after a deploy can answer 502 while the container image starts and the model loads (deploy run 31)."""

import importlib.util
import io
import json
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load("aws_verify")
call = load("call_scoring_url")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def http_error(code):
    return urllib.error.HTTPError("https://x/health", code, "x", {}, io.BytesIO(b"{}"))


class Session:
    def get_credentials(self):
        from botocore.credentials import Credentials
        return Credentials("AKIATEST", "secret")


def scripted(*answers):
    """An opener that raises or returns each answer in turn."""
    pending, seen = list(answers), []

    def opener(request, timeout):
        seen.append(request.full_url)
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(json.dumps(item).encode())

    opener.seen = seen
    return opener


class SignedHealthTests(unittest.TestCase):
    def health(self, *answers, **kwargs):
        sleeps = []
        opener = scripted(*answers)
        result = verify.signed_health(Session(), "ap-southeast-2", sleep=sleeps.append, opener=opener, **kwargs)("https://x/health")
        return result, sleeps, opener

    def test_a_502_then_a_good_answer_succeeds_after_waiting(self):
        body, sleeps, opener = self.health(http_error(502), {"model_name": "random_forest"})
        self.assertEqual(body["model_name"], "random_forest")
        self.assertEqual(sleeps, [20])
        self.assertEqual(len(opener.seen), 2)

    def test_a_timeout_is_tried_again_too(self):
        body, _, _ = self.health(TimeoutError(), {"ok": True})
        self.assertEqual(body, {"ok": True})

    def test_a_403_is_final_and_not_repeated(self):
        sleeps = []
        opener = scripted(http_error(403))
        with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            verify.signed_health(Session(), "r", sleep=sleeps.append, opener=opener)("https://x/health")
        self.assertEqual((sleeps, len(opener.seen)), ([], 1))

    def test_it_gives_up_after_the_tries_and_says_how_many(self):
        sleeps = []
        opener = scripted(*[http_error(502)] * 3)
        with self.assertRaisesRegex(RuntimeError, "HTTP 502 after 3 tries"):
            verify.signed_health(Session(), "r", retries=2, sleep=sleeps.append, opener=opener)("https://x/health")
        self.assertEqual(len(sleeps), 2)


class CallScriptTests(unittest.TestCase):
    ARGS = ["--url", "https://abc.lambda-url.ap-southeast-2.on.aws", "--region", "ap-southeast-2"]

    def run_call(self, extra, answers):
        opener = scripted(*answers)
        sleeps = []
        with mock.patch.object(call.urllib.request, "urlopen", opener), mock.patch.object(call.time, "sleep", sleeps.append), \
                mock.patch.object(call, "signed_request", lambda *a, **k: Namespace(body=None, headers={})), \
                mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            code = call.main(self.ARGS + extra)
        return code, sleeps, opener

    def test_a_get_is_retried_on_502(self):
        code, sleeps, opener = self.run_call(["--retries", "2", "GET", "/health"], [http_error(502), http_error(504), {"ok": 1}])
        self.assertEqual((code, len(opener.seen), sleeps), (0, 3, [20, 20]))

    def test_without_retries_the_first_failure_is_final(self):
        code, sleeps, opener = self.run_call(["GET", "/health"], [http_error(502)])
        self.assertEqual((code, sleeps, len(opener.seen)), (1, [], 1))

    def test_a_post_is_never_repeated(self):
        code, sleeps, opener = self.run_call(["--retries", "3", "POST", "/predict", "{}"], [http_error(502), {"ok": 1}])
        self.assertEqual((code, sleeps, len(opener.seen)), (1, [], 1))

    def test_a_403_is_not_retried(self):
        code, sleeps, _ = self.run_call(["--retries", "3", "GET", "/health"], [http_error(403)])
        self.assertEqual((code, sleeps), (1, []))

    def test_the_smoke_step_retries_health_and_posts_once(self):
        text = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
        self.assertIn("call --retries 4 GET /health", text)
        self.assertNotIn("call --retries 4 POST", text)


if __name__ == "__main__":
    unittest.main()
