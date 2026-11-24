"""Start the MLflow tracking server inside an ECS task.

The task gets the database coordinates from its environment (``DB_HOST``, ``DB_PORT``, ``DB_NAME``,
``DB_USER``) and the password from AWS Secrets Manager (``DB_PASSWORD``). This module builds the
SQLAlchemy URI (percent-encoding the user and password, which RDS-generated passwords need), hands it
to MLflow through ``MLFLOW_BACKEND_STORE_URI`` so it never appears in a process list, and starts the
server. Run artifacts are proxied to S3 (``--serve-artifacts``), so training tasks and the scoring
tools need no S3 permissions for MLflow, only the server's task role does.

``MLFLOW_ALLOWED_HOSTS`` (default ``*``) is the Host-header allow list. The default of MLflow only
accepts localhost and private addresses, and the service is reached by its Cloud Map name or by a
public IP, so it is opened here; access is limited by the security group instead.
"""

import os
import sys
from urllib.parse import quote


def backend_uri(environ) -> str:
    missing = [name for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD") if not environ.get(name)]
    if missing:
        raise ValueError(f"missing environment variable(s): {', '.join(missing)}")
    user = quote(environ["DB_USER"], safe="")
    password = quote(environ["DB_PASSWORD"], safe="")
    port = environ.get("DB_PORT", "5432")
    # The driver is named on purpose: with SQLAlchemy 2.1 a plain "postgresql://" URL means psycopg (version 3), which the
    # image does not install; it has psycopg2 (requirements/tools.txt). MLflow reads "postgresql+psycopg2" as PostgreSQL.
    return f"postgresql+psycopg2://{user}:{password}@{environ['DB_HOST']}:{port}/{environ['DB_NAME']}?sslmode=require"


def command(environ) -> list[str]:
    artifacts = environ.get("ARTIFACTS_DESTINATION")
    if not artifacts or not artifacts.startswith("s3://"):
        raise ValueError("ARTIFACTS_DESTINATION must be an s3:// URI")
    return [
        "mlflow", "server", "--host", "0.0.0.0", "--port", environ.get("PORT", "5000"),
        "--artifacts-destination", artifacts, "--serve-artifacts",
        "--allowed-hosts", environ.get("MLFLOW_ALLOWED_HOSTS", "*"),
        "--workers", environ.get("MLFLOW_WORKERS", "2"),
    ]


def main(environ=None, execvp=os.execvp) -> int:
    environ = os.environ if environ is None else environ
    try:
        uri = backend_uri(environ)
        cmd = command(environ)
    except ValueError as error:
        print(f"mlflow_server: {error}", file=sys.stderr)
        return 2
    os.environ["MLFLOW_BACKEND_STORE_URI"] = uri
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    execvp(cmd[0], cmd)
    return 0   # only reached when execvp is replaced in a test


if __name__ == "__main__":
    sys.exit(main())
