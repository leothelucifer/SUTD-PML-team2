"""Reading the scoring service's prediction log (JSON lines, gzip, Firehose batches). Needs only pandas."""

import gzip
import json
from pathlib import Path

import pandas as pd


def _json_objects(text: str):
    """Yield JSON objects from JSONL or from objects written back to back (Firehose)."""
    decoder, index = json.JSONDecoder(), 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return
        value, index = decoder.raw_decode(text, index)
        yield value


def _lines(data: bytes):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    for value in _json_objects(data.decode("utf-8")):
        if isinstance(value, dict) and "logEvents" in value:          # CloudWatch Logs -> Firehose batch
            for event in value["logEvents"]:
                try:
                    yield json.loads(event["message"])
                except (ValueError, KeyError):
                    continue
        elif isinstance(value, dict):
            yield value


def read_prediction_log(sources) -> pd.DataFrame:
    """Prediction records from files (``.jsonl``, ``.gz``, Firehose objects), folders, or raw bytes."""
    records: list[dict] = []
    for source in sources:
        if isinstance(source, bytes):
            chunks = [source]
        else:
            path = Path(source)
            files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            chunks = [file.read_bytes() for file in files]
        for chunk in chunks:
            records.extend(r for r in _lines(chunk) if r.get("record_type") == "prediction")
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    # A retried request logs the same prediction twice; count it once.
    return frame.drop_duplicates("prediction_id", keep="last").reset_index(drop=True)
