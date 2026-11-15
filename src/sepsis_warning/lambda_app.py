"""AWS Lambda entry point: the same FastAPI app as ``serve``, behind a Lambda function URL.

Configuration (environment variables, set by ``infra/cloudformation/serverless.yaml``):

* ``MODEL_S3_URI`` and ``MODEL_SHA256``: the published bundle and its pinned hash
  (from ``python -m sepsis_warning publish``). The bundle is downloaded to /tmp
  and checked before it is loaded.
* ``MODEL_PATH`` (instead of S3): a bundle already inside the container, for
  local tests with the Lambda runtime emulator.

The model loads on the first request of a new execution environment and is
reused while that environment stays warm; requests stay stateless.
"""

import os
from pathlib import Path

_adapter = None


def _build():
    from mangum import Mangum

    from .api import create_app
    from .model_source import load_serving_model

    expected = os.environ.get("MODEL_SHA256") or None
    if os.environ.get("MODEL_S3_URI"):
        service, card, source = load_serving_model(
            model_s3=os.environ["MODEL_S3_URI"], expected_sha256=expected, download_dir=Path("/tmp/model"),
        )
    elif os.environ.get("MODEL_PATH"):
        service, card, source = load_serving_model(model=Path(os.environ["MODEL_PATH"]), expected_sha256=expected)
    else:
        raise RuntimeError("set MODEL_S3_URI and MODEL_SHA256 (or MODEL_PATH for local tests)")
    print(f"Loaded {source} ({service.health()['model_name']})", flush=True)
    return Mangum(create_app(service, card), lifespan="off")


def handler(event, context):
    global _adapter
    if _adapter is None:
        _adapter = _build()
    return _adapter(event, context)
