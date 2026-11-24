"""The images step: layer cache from the previous image, no unused images, optional parallel builds. Fake aws and docker."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAKE_AWS = r'''#!/usr/bin/env bash
echo "aws $*" >> "$CALLS"
case "$1 $2" in
  "sts get-caller-identity") echo 123456789012; exit 0 ;;
  "ecr get-login-password") echo pw; exit 0 ;;
  "cloudformation describe-stacks")
    key="$(printf '%s' "$*" | grep -o "OutputKey=='[A-Za-z]*'" | head -1 | cut -d"'" -f2)"
    echo "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/repo-$(printf '%s' "$key" | tr 'A-Z' 'a-z')"; exit 0 ;;
  "ecr describe-images")
    if printf '%s' "$*" | grep -q -- "--image-ids"; then
      for tag in $EXISTING_TAGS; do printf '%s' "$*" | grep -q "imageTag=$tag" && { echo sha256:abc; exit 0; }; done
      exit 254
    fi
    if [ "$NO_PREVIOUS" = 1 ]; then echo None; exit 0; fi
    kind="$(printf '%s' "$*" | grep -o "starts_with(imageTags\[0\], '[a-z]*-')" | cut -d"'" -f2)"
    echo "${kind}old1"; exit 0 ;;
esac
exit 0
'''

FAKE_DOCKER = r'''#!/usr/bin/env bash
echo "docker $*" >> "$CALLS"
if [ "$1" = login ]; then cat > /dev/null; exit 0; fi
if [ "$1" = build ] && [ -n "$FAIL_KIND" ] && printf '%s' "$*" | grep -q -- "--target $FAIL_KIND "; then echo "build of $FAIL_KIND broke" >&2; exit 1; fi
exit 0
'''


def run(**env):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for name, body in (("aws", FAKE_AWS), ("docker", FAKE_DOCKER)):
            path = temp / name
            path.write_text(body)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        calls = temp / "calls.log"
        full = {**os.environ, "PATH": f"{temp}:{os.environ['PATH']}", "AWS_REGION": "ap-southeast-2", "ENVIRONMENT": "dev",
                "TAG": "abc123", "CALLS": str(calls), "EXISTING_TAGS": "", "NO_PREVIOUS": "0", "FAIL_KIND": "", **env}
        result = subprocess.run(["bash", str(ROOT / "scripts" / "aws_deploy.sh"), "images"], env=full, capture_output=True,
                                text=True, cwd=ROOT)
        return result, calls.read_text() if calls.exists() else ""


def builds(calls: str) -> list[str]:
    return [line for line in calls.splitlines() if line.startswith("docker build")]


class ImagesStepTests(unittest.TestCase):
    def test_lambda_mode_builds_three_images_and_skips_the_fargate_serving_image(self):
        result, calls = run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Skipping the scoring image", result.stdout)
        targets = [line.split("--target ")[1].split()[0] for line in builds(calls)]
        self.assertEqual(sorted(targets), ["lambda", "tools", "training"])
        self.assertEqual(calls.count("docker push --quiet"), 3)

    def test_fargate_serving_mode_builds_all_four(self):
        result, calls = run(SERVING_MODE="fargate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(builds(calls)), 4)

    def test_each_build_reuses_the_layers_of_the_previous_image_of_the_same_kind(self):
        result, calls = run()
        for line in builds(calls):
            kind = line.split("--target ")[1].split()[0]
            self.assertIn(f":{kind}-old1", line.split("--cache-from ")[1].split()[0])
            self.assertIn("BUILDKIT_INLINE_CACHE=1", line)               # so the next build can reuse these layers too
            self.assertIn("docker pull --quiet", calls)
        self.assertIn("layers reused from", result.stdout)

    def test_the_first_build_ever_has_nothing_to_reuse_and_still_works(self):
        result, calls = run(NO_PREVIOUS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--cache-from", calls)
        self.assertIn("no earlier image to reuse", result.stdout)

    def test_an_image_that_exists_for_this_commit_is_reused_not_rebuilt(self):
        result, calls = run(EXISTING_TAGS="training-abc123 tools-abc123")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Reusing existing training image", result.stdout)
        self.assertEqual([line.split("--target ")[1].split()[0] for line in builds(calls)], ["lambda"])

    def test_parallel_builds_run_every_kind_and_a_failure_fails_the_step_with_its_log(self):
        ok, calls = run(IMAGES_PARALLEL="1")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(len(builds(calls)), 3)
        for kind in ("lambda", "training", "tools"):
            self.assertIn(f"---- {kind} ----", ok.stdout)
        bad, _ = run(IMAGES_PARALLEL="1", FAIL_KIND="training")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("build of training broke", bad.stdout + bad.stderr)
        self.assertIn("an image build failed", bad.stderr + bad.stdout)

    def test_the_dockerfile_installs_dependencies_before_it_copies_the_source(self):
        text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        for target in ("training", "tools", "lambda", "scoring"):
            stage = text.split(f"FROM base AS {target}", 1)[1].split("\nFROM ", 1)[0]
            self.assertLess(stage.index(f"requirements/{target}.txt"), stage.index("COPY src/sepsis_warning"), target)
            self.assertLess(stage.index("--require-hashes"), stage.index("COPY src/sepsis_warning"), target)
        base = text.split("FROM ${BASE_IMAGE} AS base", 1)[1].split("\nFROM ", 1)[0]
        self.assertNotIn("COPY src", base)                                   # a code change must not touch the shared layers


if __name__ == "__main__":
    unittest.main()
