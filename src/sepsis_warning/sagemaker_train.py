"""SageMaker AI training job entry point: ``train`` inside the standard /opt/ml layout.

SageMaker starts the training image with the argument ``train``. This module:

1. reads hyperparameters from ``/opt/ml/input/config/hyperparameters.json``
   (all values arrive as strings): ``seed``, ``window``, ``budget``,
   ``cooldown``, ``mlflow_tracking_uri`` (the MLflow App ARN),
   ``register_model`` ("true"/"false"), ``git_commit``;
2. reads the ``data`` channel, ``/opt/ml/input/data/data/``: either
   ``combined.parquet`` with ``splits.csv`` (and optionally
   ``splits.csv.summary.json`` and ``dvc.lock``), or a folder with
   ``training_setA``/``training_setB`` (the synthetic fixture);
3. runs the same ``run_experiment`` as ``python -m sepsis_warning train``:
   every candidate is fitted, selection uses the validation split only;
4. writes the run to ``/opt/ml/model``, which SageMaker uploads to the
   encrypted artifact bucket as ``model.tar.gz``, and, when a tracking URI is
   given, logs the selection run and registers the bundle as ``@candidate``.

On failure the reason goes to ``/opt/ml/output/failure``, which SageMaker
shows as the job's failure reason. ``SM_ROOT`` overrides ``/opt/ml`` for tests.
"""

import json
import os
import sys
import traceback
from argparse import Namespace
from pathlib import Path


def _root() -> Path:
    return Path(os.environ.get("SM_ROOT", "/opt/ml"))


def _flag(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def build_args(root: Path) -> Namespace:
    config = root / "input" / "config" / "hyperparameters.json"
    params = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
    channel = root / "input" / "data" / "data"
    if (channel / "combined.parquet").is_file():
        data, splits = channel / "combined.parquet", channel / "splits.csv"
        if not splits.is_file():
            raise ValueError("the data channel has combined.parquet but no splits.csv; upload the team's saved split")
    elif (channel / "training_setA").is_dir():
        data, splits = channel, None
    else:
        raise ValueError(f"no combined.parquet or training_setA in the data channel {channel}")
    summary = channel / "splits.csv.summary.json"
    dvc_lock = channel / "dvc.lock"
    tracking = params.get("mlflow_tracking_uri") or None
    if params.get("git_commit"):
        os.environ["SEPSIS_GIT_COMMIT"] = params["git_commit"]
    return Namespace(
        data=data, splits=splits, output=root / "model",
        seed=int(params.get("seed", 42)), window=int(params.get("window", 6)),
        budget=float(params.get("budget", 0.2)), cooldown=int(params.get("cooldown", 6)),
        splits_summary=summary if summary.is_file() else None,
        dvc_lock=dvc_lock if dvc_lock.is_file() else None,
        mlflow_uri=tracking, register_model=_flag(params.get("register_model", "false")) and tracking is not None,
    )


def main(argv=None) -> int:
    root = _root()
    try:
        from .cli import run_experiment
        from .tracking import check_tracking_uri, require_mlflow

        args = build_args(root)
        if args.mlflow_uri:
            check_tracking_uri(args.mlflow_uri)
            require_mlflow()
        print(f"Training on {args.data} (seed {args.seed}); outputs to {args.output}", flush=True)
        run_experiment(args)
        return 0
    except Exception as error:  # noqa: BLE001 - SageMaker needs the reason in a file
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "output" / "failure").write_text(f"{type(error).__name__}: {error}"[:1000], encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
