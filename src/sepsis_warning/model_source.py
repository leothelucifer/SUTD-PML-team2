"""Move frozen bundles between the registry, S3 and a serving process (Lambda or Fargate).

* ``publish_model`` copies a registered version's bundle and its validation
  summary to ``s3://<bucket>/models/v<version>/`` and returns the S3 URI and
  SHA-256 that a deployment pins.
* ``fetch_model`` downloads a bundle from S3 and refuses it unless its SHA-256
  matches the pinned value, so a changed object is never loaded. Joblib files
  can run code when loaded; the hash is checked before anything is unpickled.

Needs ``boto3`` (the ``aws`` extra; Lambda's Python runtime already has it).
The bucket's default KMS encryption applies to every upload.
"""

import json
import tempfile
from pathlib import Path

from .scoring import file_sha256

BUNDLE_FILE = "model.joblib"
CARD_FILES = ("selection_metrics.json", "run_metadata.json", "manifest.json", "registry.json")


class ModelHashMismatch(ValueError):
    """The object in S3 is not the bundle the deployment pinned."""


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://") or "/" not in uri[5:]:
        raise ValueError(f"expected s3://bucket/key, got {uri!r}")
    bucket, key = uri[5:].split("/", 1)
    if not bucket or not key:
        raise ValueError(f"expected s3://bucket/key, got {uri!r}")
    return bucket, key


def _s3(client):
    if client is not None:
        return client
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("S3 access needs boto3; run `pip install -e .[aws]`") from error
    return boto3.client("s3")


def _missing(error) -> bool:
    code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")


def fetch_model(model_s3_uri: str, expected_sha256: str, destination: Path, client=None) -> Path:
    """Download the bundle (and any card files beside it) and check the bundle's hash."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("a 64-character SHA-256 for the bundle is required")
    bucket, key = parse_s3_uri(model_s3_uri)
    s3 = _s3(client)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / BUNDLE_FILE
    s3.download_file(bucket, key, str(target))
    actual = file_sha256(target)
    if actual != expected_sha256:
        target.unlink()
        raise ModelHashMismatch(f"{model_s3_uri} has SHA-256 {actual[:12]}…, expected {expected_sha256[:12]}…")
    prefix = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
    for name in CARD_FILES:
        try:
            s3.download_file(bucket, prefix + name, str(destination / name))
        except Exception as error:
            if not _missing(error):
                raise
    return target


def publish_model(model_uri: str, tracking_uri: str, bucket: str, prefix: str = "models", client=None) -> dict:
    """Copy a registered version to S3 for deployment. Returns what to pin."""
    from .registry import resolve_model

    s3 = _s3(client)
    with tempfile.TemporaryDirectory(prefix="sepsis-publish-") as temp:
        path, sha, tags = resolve_model(model_uri, tracking_uri, Path(temp))
        version = tags["version"]
        base = f"{prefix.strip('/')}/v{version}"
        registry = {
            "model_uri": model_uri, "version": version,
            "reviewed_by": tags.get("production_reviewed_by"), "promoted_at": tags.get("production_promoted_at"),
        }
        (path.parent / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
        s3.upload_file(str(path), bucket, f"{base}/{BUNDLE_FILE}",
                       ExtraArgs={"Metadata": {"sha256": sha, "model-version": str(version)}})
        for name in CARD_FILES:
            if (path.parent / name).is_file():
                s3.upload_file(str(path.parent / name), bucket, f"{base}/{name}")
    return {"model_s3_uri": f"s3://{bucket}/{base}/{BUNDLE_FILE}", "sha256": sha, "version": str(version)}


def load_serving_model(*, model: Path | None = None, model_uri: str | None = None, model_s3: str | None = None,
                       expected_sha256: str | None = None, mlflow_uri: str | None = None,
                       download_dir: Path | None = None, client=None, allow_unverified: bool = False):
    """Load one bundle for serving, from a file, the registry or S3. Returns (service, card, source).

    Shared by ``serve`` and the Lambda handler so every path checks the hash and
    builds the same model card. A local file is verified against the given hash or
    the one in ``manifest.json`` beside it.
    """
    from .model_card import card_from_run_dir
    from .service import ScoringService

    if sum(x is not None for x in (model, model_uri, model_s3)) != 1:
        raise ValueError("give exactly one of a model file, a registry URI or an S3 URI")
    registry = None
    if model_uri:
        from .registry import resolve_model

        path, expected, tags = resolve_model(model_uri, mlflow_uri, download_dir)
        registry = {"model_uri": model_uri, "version": tags["version"],
                    "reviewed_by": tags.get("production_reviewed_by"), "promoted_at": tags.get("production_promoted_at")}
        source = f"{model_uri} (version {tags['version']})"
    elif model_s3:
        if not expected_sha256:
            raise ValueError("--model-s3 needs --model-sha256: an S3 object is only loaded with a pinned hash")
        target = Path(download_dir or tempfile.mkdtemp(prefix="sepsis-model-"))
        path, expected = fetch_model(model_s3, expected_sha256, target, client), expected_sha256
        source = model_s3
    else:
        from .manifest import recorded_sha256

        path, source = Path(model), str(model)
        expected = expected_sha256 or recorded_sha256(path)
        if expected is None and not allow_unverified:
            raise ValueError(
                f"{path} has no manifest.json beside it and no SHA-256 was given; pass the hash "
                "(--model-sha256), or --allow-unverified-model for a bundle you built yourself"
            )
    service = ScoringService.from_path(path, expected)
    card = card_from_run_dir(Path(path).parent, service.bundle, registry)
    return service, card, source
