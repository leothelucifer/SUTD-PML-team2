"""Run one pipeline step as an ECS (Fargate) task: the same code as the SageMaker training job.

SageMaker starts ``sagemaker_train`` with its own folder layout (``/opt/ml``), downloads the
input channels and uploads ``/opt/ml/model`` afterwards. On ECS none of that exists, so this
module does those three things and then calls the unchanged entry point:

1. ``HP_<name>=<value>`` environment variables become ``hyperparameters.json``
   (``HP_mode=tune``, ``HP_trials=6``, ...).
2. ``CH_<channel>=s3://bucket/prefix/`` variables name the input channels; every object under
   the prefix is copied to ``input/data/<channel>/``.
3. ``sagemaker_train.main()`` runs against that folder (``SM_ROOT``).
4. ``JOB_OUTPUT=s3://bucket/prefix/`` receives ``model/...`` (what SageMaker would archive) and,
   if the step failed, a ``failure`` file with the reason. The exit code is the step's.

Only the pipeline's task role may write ``JOB_OUTPUT``; the image holds no data or credentials.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

HP_PREFIX = "HP_"
CHANNEL_PREFIX = "CH_"
S3_CLIENT = None  # tests set a stand-in; in a task boto3 uses the task role


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _client(client=None):
    client = client or S3_CLIENT
    if client is None:
        import boto3

        client = boto3.client("s3")
    return client


def hyperparameters(environ) -> dict:
    """``HP_name=value`` variables as the string-valued dictionary SageMaker would write."""
    return {key[len(HP_PREFIX):]: value for key, value in sorted(environ.items()) if key.startswith(HP_PREFIX)}


def channels(environ) -> dict:
    return {key[len(CHANNEL_PREFIX):]: value for key, value in sorted(environ.items()) if key.startswith(CHANNEL_PREFIX)}


def download_prefix(uri: str, target: Path, client=None) -> list[str]:
    """Copy every object under ``uri`` to ``target`` (keeping relative paths). Returns the relative paths."""
    client = _client(client)
    bucket, prefix = parse_s3_uri(uri)
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    copied = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("Contents", []):
            relative = item["Key"][len(prefix):]
            if not relative or relative.endswith("/"):
                continue
            destination = target / relative
            if target.resolve() not in destination.resolve().parents:
                raise ValueError(f"object key escapes the channel folder: {item['Key']!r}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, item["Key"], str(destination))
            copied.append(relative)
        if not page.get("IsTruncated"):
            return copied
        token = page.get("NextContinuationToken")


def upload_directory(directory: Path, uri: str, client=None) -> list[str]:
    client = _client(client)
    bucket, prefix = parse_s3_uri(uri)
    uploaded = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            key = f"{prefix.rstrip('/')}/{path.relative_to(directory).as_posix()}".lstrip("/")
            client.upload_file(str(path), bucket, key)
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def stage(root: Path, environ, client=None) -> None:
    """Build the ``/opt/ml``-style folder for one step."""
    import json

    (root / "input" / "config").mkdir(parents=True, exist_ok=True)
    (root / "input" / "config" / "hyperparameters.json").write_text(
        json.dumps(hyperparameters(environ)), encoding="utf-8")
    for name, uri in channels(environ).items():
        files = download_prefix(uri, root / "input" / "data" / name, client)
        print(f"Channel {name}: {len(files)} file(s) from {uri}", flush=True)


def run(environ=None, client=None, workdir: Path | None = None) -> int:
    environ = os.environ if environ is None else environ
    output = environ.get("JOB_OUTPUT")
    if not output:
        print("JOB_OUTPUT is not set", file=sys.stderr)
        return 2
    parse_s3_uri(output)
    own_tmp = workdir is None
    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="sepsis-ecs-"))
    try:
        try:
            stage(root, environ, client)
        except Exception as error:  # noqa: BLE001 - report the reason the same way a failed step would
            (root / "output").mkdir(parents=True, exist_ok=True)
            (root / "output" / "failure").write_text(f"{type(error).__name__}: {error}"[:1000], encoding="utf-8")
            code = 1
        else:
            os.environ["SM_ROOT"] = str(root)
            from . import sagemaker_train

            code = sagemaker_train.main([])
        for folder in ("model", "output"):
            if (root / folder).is_dir():
                upload_directory(root / folder, output.rstrip("/") + f"/{folder}/", client)
        return code
    finally:
        if own_tmp:
            shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
