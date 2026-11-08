"""Save the patient split once (splits.csv) so every teammate uses the same one.

    python -m sepsis_warning split --data data/physionet/combined.parquet --output data/physionet/splits.csv

The split is by ICU stay (one .psv record). The dataset has no person ID, so
the same person could have two stays in different splits; record-level
separation does not prove person-level separation.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .data import Patient, split_patients

SPLIT_ORDER = ["train", "calibration", "validation", "internal_test", "external_test"]


def split_table(parquet: Path, seed: int) -> pd.DataFrame:
    """One row per ICU stay: patient_id, hospital, septic, split."""
    labels = pd.read_parquet(parquet, columns=["hospital", "patient_id", "SepsisLabel"])
    stays = [
        Patient(f"{hospital}:{pid}", hospital, frame[["SepsisLabel"]])
        for (hospital, pid), frame in labels.groupby(["hospital", "patient_id"], sort=True)
    ]
    rows = [
        {"patient_id": p.patient_id.split(":", 1)[1], "hospital": p.hospital, "septic": int(p.septic), "split": name}
        for name, cohort in split_patients(stays, seed).items() for p in cohort
    ]
    table = pd.DataFrame(rows)
    table["split"] = pd.Categorical(table["split"], SPLIT_ORDER, ordered=True)
    return table.sort_values(["split", "hospital", "patient_id"]).reset_index(drop=True)


def write_splits(parquet: Path, output: Path, seed: int) -> dict:
    table = split_table(parquet, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    counts = table.groupby("split", observed=True).agg(stays=("patient_id", "size"), septic=("septic", "sum"))
    summary = {
        "seed": seed,
        "unit": "ICU stay (.psv record); not guaranteed person-disjoint",
        "source_parquet": str(parquet),
        "source_parquet_sha256": _sha256(parquet),
        "splits_csv_sha256": _sha256(output),
        "counts": {name: {"stays": int(r.stays), "septic": int(r.septic)} for name, r in counts.iterrows()},
    }
    output.with_name(f"{output.name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_splits(path: Path) -> dict[str, str]:
    """Map 'A:p000001' style IDs to split names."""
    table = pd.read_csv(path, dtype=str)
    return dict(zip(table["hospital"] + ":" + table["patient_id"], table["split"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
