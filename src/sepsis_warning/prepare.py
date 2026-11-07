"""Combine validated PhysioNet PSV records into one local Parquet file."""

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import read_patient

SOURCE_URL = "https://physionet.org/content/challenge-2019/1.0.0/"
BATCH_RECORDS = 128


def prepare_parquet(data_root: Path, output: Path) -> dict:
    """Validate and combine A/B PSV records; never overwrite existing output."""
    data_root = data_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new output path")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"{temporary} already exists; inspect before retrying")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    expected_columns = None
    rows = 0
    file_counts = {"A": 0, "B": 0}
    row_counts = {"A": 0, "B": 0}
    batch: list[pd.DataFrame] = []
    try:
        for hospital in ("A", "B"):
            files = sorted((data_root / f"training_set{hospital}").glob("*.psv"))
            if not files:
                raise ValueError(
                    f"No patient PSV records found for source {hospital} under {data_root}"
                )
            for path in files:
                frame = read_patient(path).astype("float64")
                columns = tuple(frame.columns)
                if expected_columns is None:
                    expected_columns = columns
                elif columns != expected_columns:
                    raise ValueError(f"{path}: PSV schema differs from earlier records")
                frame.insert(0, "patient_id", path.stem)
                frame.insert(1, "hospital", hospital)
                batch.append(frame)
                file_counts[hospital] += 1
                row_counts[hospital] += len(frame)
                rows += len(frame)

                if len(batch) >= BATCH_RECORDS:
                    writer = _write_batch(batch, temporary, writer)
                    batch.clear()

        if batch:
            writer = _write_batch(batch, temporary, writer)
            batch.clear()
        if writer is None:
            raise ValueError(f"No patient PSV records found under {data_root}")
        writer.close()
        writer = None

        parquet_rows = pq.ParquetFile(temporary).metadata.num_rows
        if parquet_rows != rows:
            raise ValueError(
                f"Parquet row count mismatch: wrote {parquet_rows}, expected {rows}"
            )
        if output.exists():
            raise FileExistsError(
                f"{output} appeared while preparing; refusing to overwrite it"
            )
        os.rename(temporary, output)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    return {
        "source": SOURCE_URL,
        "format": "parquet",
        "output": str(output),
        "records": sum(file_counts.values()),
        "hourly_rows": rows,
        "records_by_hospital": file_counts,
        "hourly_rows_by_hospital": row_counts,
        "columns": ["patient_id", "hospital", *(expected_columns or ())],
    }


def _write_batch(batch: list[pd.DataFrame], path: Path, writer):
    table = pa.Table.from_pandas(pd.concat(batch, ignore_index=True), preserve_index=False)
    if writer is None:
        schema = table.schema.with_metadata({
            b"dataset_source": SOURCE_URL.encode(),
            b"patient_id_note": b"PSV stem; grouping key only, not a model feature",
        })
        writer = pq.ParquetWriter(path, schema, compression="zstd")
    writer.write_table(table, row_group_size=50_000)
    return writer


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
