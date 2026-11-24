"""Register a local tabular file and replay its original rows in bounded batches."""

import csv
import json
import re
import shutil
import tempfile
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
from openpyxl import load_workbook


_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def _numeric_kind(value):
    """Infer common numeric types conservatively, preserving padded IDs as text."""
    if _INTEGER.fullmatch(value):
        if len(value) <= 20 and -(2**63) <= int(value) < 2**63:
            return "Int64"
    elif _DECIMAL.fullmatch(value) and abs(float(value)) < float("inf"):
        return "Float64"
    return "string"


@contextmanager
def _rows(path, settings):
    if path.suffix == ".csv":
        with path.open(encoding=settings["encoding"], newline="") as stream:
            yield csv.reader(stream, delimiter=settings["delimiter"], strict=True)
    else:
        book = load_workbook(path, read_only=True, data_only=False)
        try:
            yield book[settings["sheet"]].values
        finally:
            book.close()


def _settings(path, delimiter, sheet, encoding, header_row):
    if header_row < 1:
        raise ValueError("header_row starts at 1")
    if path.suffix == ".csv":
        if sheet is not None:
            raise ValueError("sheet applies only to Excel files")
        if delimiter is None:
            with path.open(encoding=encoding, newline="") as stream:
                line = next(islice(stream, header_row - 1, header_row), "")
            candidates = [char for char in ",;\t|" if
                          len(next(csv.reader([line], delimiter=char), [])) > 1]
            if len(candidates) != 1:
                raise ValueError("Specify delimiter explicitly for ambiguous or single-column CSV")
            delimiter = candidates[0]
        if len(delimiter) != 1 or delimiter in "\r\n":
            raise ValueError("Choose a single delimiter character")
    else:
        if delimiter is not None:
            raise ValueError("delimiter applies only to CSV files")
        book = load_workbook(path, read_only=True)
        try:
            names = [ws.title for ws in book.worksheets]
            if sheet is None and len(names) == 1:
                sheet = names[0]
            if sheet not in names:
                raise ValueError(f"Choose a worksheet from {names}")
        finally:
            book.close()
    return dict(delimiter=delimiter, sheet=sheet, encoding=encoding, header_row=header_row)


def load_dataset(source, storage_dir, *, delimiter=None, sheet=None,
                 encoding="utf-8-sig", header_row=1):
    """Copy, validate and register one table. Return its JSON-compatible manifest."""
    source = Path(source)
    suffix = source.suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise ValueError("Supported formats are .csv and .xlsx")
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid4())
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        staged = Path(temporary) / dataset_id
        staged.mkdir()
        path = staged / ("source" + suffix)
        shutil.copyfile(source, path)
        settings = _settings(path, delimiter, sheet, encoding, header_row)
        with _rows(path, settings) as rows:
            rows = iter(rows)
            header = next(islice(rows, header_row - 1, header_row), ())
            columns = [str(name) if name is not None else "" for name in header]
            if not columns or any(not name.strip() for name in columns):
                raise ValueError("Select a header row containing a name for every column")
            if len(set(columns)) != len(columns):
                raise ValueError("Column headers must be unique")
            kinds = [None] * len(columns)
            preview = []
            count = 0
            for count, row in enumerate(rows, 1):
                if len(row) != len(columns):
                    raise ValueError(f"Data record {count} has {len(row)} fields, expected {len(columns)}")
                if count <= 5:
                    preview.append(list(row))
                if suffix == ".csv":
                    for i, value in enumerate(row):
                        if value and kinds[i] != "string":
                            kind = _numeric_kind(value)
                            if kind == "Float64" or kinds[i] is None or kind == "string":
                                kinds[i] = kind
        manifest = dict(dataset_id=dataset_id, version=1, source_name=source.name,
                        format=suffix[1:], settings=settings, columns=columns,
                        row_count=count, preview=preview, warnings=[],
                        csv_dtypes=[k or "string" for k in kinds] if suffix == ".csv" else None)
        serialized = json.dumps(manifest, default=str, ensure_ascii=False)
        (staged / "manifest.json").write_text(serialized, encoding="utf-8")
        staged.rename(root / dataset_id)
    return json.loads(serialized)


def iter_dataset(storage_dir, dataset_id, *, batch_size=50000):
    """Yield DataFrames in source order. Even a header-only table yields its schema."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    directory = Path(storage_dir) / str(UUID(dataset_id))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    settings = manifest["settings"]
    path = directory / ("source." + manifest["format"])
    with _rows(path, settings) as rows:
        rows = islice(rows, settings["header_row"], None)
        first = True
        while True:
            batch = list(islice(rows, batch_size))
            if not batch and not first:
                break
            frame = pd.DataFrame(batch, columns=manifest["columns"])
            if manifest["csv_dtypes"]:
                for name, dtype in zip(frame, manifest["csv_dtypes"]):
                    series = frame[name].mask(frame[name].eq(""), pd.NA)
                    frame[name] = series.astype(dtype)
            yield frame
            first = False
            if len(batch) < batch_size:
                break
