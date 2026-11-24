#!/usr/bin/env bash
# Regenerate the hash-locked requirement files that the images and CI install.
#
#   bash scripts/lock_requirements.sh          # needs `pip install uv`
#
# requirements/<target>.txt lists every package (direct and transitive) with its
# exact version and SHA-256, for Python 3.12 on Linux x86_64, which is what the
# images and CI run. They are installed with `pip --require-hashes`, so two builds
# of the same commit are identical and a tampered package is refused.
#
# The scientific versions come from requirements.txt (the versions behind the
# reported results). python-dateutil is left out of that constraint: its pin there
# conflicts with botocore, which the AWS images need.
# MLFLOW_VERSION pins the training image to the SageMaker MLflow App's version.
set -euo pipefail
cd "$(dirname "$0")/.."
MLFLOW_VERSION="${MLFLOW_VERSION:-3.17.2}"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
grep -v '^python-dateutil\|^#' requirements.txt > "$work/base.txt"

lock() {  # lock NAME "extra1 extra2 ..." [extra constraint line]
  local name="$1" extras=() ; cp "$work/base.txt" "$work/$name.txt"
  # Evidently (the "tools" image) does not import under pandas 3, which the training stack pins; the tools image
  # only reads prediction logs and reference tables, so it takes pandas 2 and never loads a model bundle.
  [ "$name" = tools ] && { grep -v '^pandas==' "$work/$name.txt" > "$work/$name.tmp"; mv "$work/$name.tmp" "$work/$name.txt"; echo "pandas<3" >> "$work/$name.txt"; }
  [ -n "${3:-}" ] && echo "$3" >> "$work/$name.txt"
  for extra in $2; do extras+=(--extra "$extra"); done
  uv pip compile pyproject.toml "${extras[@]}" -c "$work/$name.txt" --generate-hashes \
    --python-version 3.12 --python-platform x86_64-manylinux_2_28 --no-header -q -o "requirements/$name.txt"
  # The constraint file lives in a temporary folder; keep that path out of the committed lock file.
  sed -i -E "s#-c [^ ]*/([A-Za-z0-9_.-]+\\.txt)#-c constraints/\\1#" "requirements/$name.txt"
  echo "requirements/$name.txt: $(grep -c '==' "requirements/$name.txt") packages"
}

lock scoring  "serve aws"
lock lambda   "serve lambda"
lock training "tracking aws sagemaker checks" "mlflow==$MLFLOW_VERSION"
lock tools    "tools" "mlflow==$MLFLOW_VERSION"
lock ci       "serve test tracking aws lambda checks"
