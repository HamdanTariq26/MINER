import csv
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, time, timedelta
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

from .loader import _rows, load_dataset


def _validate(steps, columns):
    """Check the complete request before writing any output."""
    if not isinstance(steps, list) or not steps:
        raise ValueError("Supply a nonempty list of cleaning steps")

    extra_arguments = {
        "trim": set(),
        "mark_missing": {"markers"},
        "map_categories": {"mapping"},
        "drop_duplicates": set(),
    }
    compiled = []

    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each step must be an object")

        operation = step.get("operation")
        if not isinstance(operation, str) or operation not in extra_arguments:
            raise ValueError(f"Unknown cleaning operation: {operation}")

        expected = {"operation", "columns"} | extra_arguments[operation]
        if set(step) != expected:
            raise ValueError(
                f"{operation} expects these arguments: {sorted(expected)}"
            )

        names = step["columns"]
        if not isinstance(names, list) or not names:
            raise ValueError(f"{operation} needs a nonempty column list")

        if any(not isinstance(name, str) or name not in columns for name in names):
            raise ValueError(f"Unknown column in {names}")

        if len(set(names)) != len(names):
            raise ValueError("Repeated column name")

        argument = None

        if operation == "mark_missing":
            markers = step["markers"]
            if not isinstance(markers, list) or not all(
                isinstance(value, str) for value in markers
            ):
                raise ValueError("markers must be a list of strings")
            argument = set(markers)

        if operation == "map_categories":
            argument = step["mapping"]
            if not isinstance(argument, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in argument.items()
            ):
                raise ValueError("mapping must map strings to strings")

        indices = [columns.index(name) for name in names]
        compiled.append((operation, indices, argument))

    return compiled


@contextmanager
def _writer(path, sheet):
    """Yield a row-writing function for CSV or a single Excel worksheet."""
    if path.suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as output:
            yield csv.writer(output).writerow
    else:
        book = Workbook(write_only=True)
        worksheet = book.create_sheet(sheet)

        def append(row):
            cells = []
            for value in row:
                cell = WriteOnlyCell(worksheet, value=value)
                if isinstance(value, str):
                    cell.data_type = "s"
                elif value is None:
                    # Preserve trailing empty cells when the loader reopens XLSX.
                    cell.value = ""
                cells.append(cell)
            worksheet.append(cells)

        try:
            yield append
        finally:
            book.save(path)
            book.close()


def _key_value(value):
    """Represent source values consistently in a duplicate key."""
    if value is None or value == "":
        return None
    if isinstance(value, (date, time)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, timedelta):
        return ["duration", value.days, value.seconds, value.microseconds]
    return value


def clean_dataset(storage_dir, dataset_id, steps):
    """Apply explicitly requested operations and register a derived dataset."""
    root = Path(storage_dir)
    parent = root / str(UUID(dataset_id))
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))

    columns = manifest["columns"]
    compiled = _validate(steps, columns)
    settings = manifest["settings"]
    suffix = "." + manifest["format"]

    changes = [
        {"operation": operation, "cells_changed": 0, "rows_removed": 0}
        for operation, _, _ in compiled
    ]

    with TemporaryDirectory(dir=root) as temporary:
        work = Path(temporary)
        output = work / ("cleaned" + suffix)
        source = parent / ("source" + suffix)

        # Store duplicate keys on disk so duplicates can be found across
        # the whole file. Each deduplication step has its own set of keys.
        with closing(sqlite3.connect(work / "seen.sqlite")) as db:
            db.execute(
                "CREATE TABLE seen (step INTEGER, key TEXT, "
                "PRIMARY KEY(step, key)) WITHOUT ROWID"
            )

            with _rows(source, settings) as rows, _writer(output, settings["sheet"]) as write:
                write(columns)

                for raw in islice(rows, settings["header_row"], None):
                    row = list(raw)

                    for i, (operation, indices, argument) in enumerate(compiled):
                        if operation == "drop_duplicates":
                            key = json.dumps([_key_value(row[j]) for j in indices])
                            inserted = db.execute(
                                "INSERT OR IGNORE INTO seen VALUES (?, ?)",
                                (i, key),
                            )
                            if inserted.rowcount == 0:
                                changes[i]["rows_removed"] += 1
                                break
                        else:
                            for j in indices:
                                old = row[j]
                                if not isinstance(old, str):
                                    continue

                                if operation == "trim":
                                    new = old.strip()
                                elif operation == "mark_missing":
                                    new = "" if old in argument else old
                                else:
                                    new = argument.get(old, old)

                                if new != old:
                                    row[j] = new
                                    changes[i]["cells_changed"] += 1
                    else:
                        # Runs when all steps finished without dropping the row.
                        write(row)

        # Register privately first. Publish only after the data, manifest,
        # and parent/version information have all been written successfully.
        options = {"delimiter": ","} if suffix == ".csv" else {"sheet": settings["sheet"]}
        child = load_dataset(output, work, **options)

        result = {
            "status": "ok",
            "tool": "clean_dataset",
            "dataset_id": child["dataset_id"],
            "parent_dataset_id": manifest["dataset_id"],
            "version": manifest["version"] + 1,
            "rows_before": manifest["row_count"],
            "rows_after": child["row_count"],
            "changes": changes,
        }

        child.update(
            version=result["version"],
            parent_dataset_id=result["parent_dataset_id"],
            cleaning_steps=steps,
            cleaning_result=result,
        )

        folder = work / child["dataset_id"]
        (folder / "manifest.json").write_text(
            json.dumps(child), encoding="utf-8"
        )
        folder.rename(root / child["dataset_id"])

    return result