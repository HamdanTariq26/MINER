"""Profile a registered MINER dataset in batches.

The loader supplies pandas DataFrames. SQLite stores the rows temporarily
so distinct counts, duplicates, quartiles, and frequencies can be calculated
across the entire dataset. No sampling is performed.

The returned dictionary follows the structure of profiler.profile_dataset().
"""

import datetime as dt
import math
import sqlite3
import statistics
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .loader import iter_dataset
from .profiler import _safe_number, profile_dataset


def count_rows_and_missing(batches):
    """Count rows and missing values without storing the dataset."""
    rows = 0
    missing = {}

    for batch in batches:
        rows += len(batch)

        for name, count in batch.isna().sum().items():
            missing[name] = missing.get(name, 0) + int(count)

    return {"rows": rows, "missing_counts": missing}


def _schema(read_batches):
    """Determine a common dtype for each column across all batches.

    CSV batches from the loader already have consistent dtypes. Excel
    batches can differ, for example when one batch contains only numbers
    and a later batch contains text in the same column.
    """
    examples = None

    for batch in read_batches():
        if examples is None:
            examples = {name: {} for name in batch.columns}

        if list(examples) != list(batch.columns) or not batch.columns.is_unique:
            raise ValueError("All batches must have the same unique columns")

        for name, dtype in batch.dtypes.items():
            if dtype not in examples[name]:
                examples[name][dtype] = batch[name].iloc[:1].copy()

    if examples is None:
        raise ValueError("An empty dataset must still yield its column schema")

    return pd.DataFrame({
        name: pd.concat(witnesses.values(), ignore_index=True).iloc[:0]
        for name, witnesses in examples.items()
    })


def _store(value):
    """Convert a loader value into a SQLite-compatible value."""
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        value = value.item()

    # Tag temporal values so an Excel date remains distinguishable
    # from an ordinary text cell containing the same date.
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        tag = type(value).__name__
        return (tag + "|" + value.isoformat()).encode()

    if isinstance(value, dt.timedelta):
        nanoseconds = pd.Timedelta(value).value
        return ("Timedelta|" + str(nanoseconds)).encode()

    return value


def _restore(value):
    """Restore temporal values when returning categorical frequencies."""
    if not isinstance(value, bytes):
        return value

    tag, data = value.decode().split("|", 1)

    if tag == "Timedelta":
        return pd.Timedelta(int(data), unit="ns")

    parsers = {
        "Timestamp": pd.Timestamp,
        "datetime": dt.datetime.fromisoformat,
        "time": dt.time.fromisoformat,
        "date": dt.date.fromisoformat,
    }
    return parsers[tag](data)


def _stage(db, read_batches, schema, fields):
    """Write typed rows to disk and count rows and DataFrame memory."""
    db.execute(f"CREATE TABLE observations ({', '.join(fields)})")

    placeholders = ", ".join("?" for _ in fields)
    insert = f"INSERT INTO observations VALUES ({placeholders})"

    rows = 0
    memory = 0
    dtypes = schema.dtypes.to_dict()

    for frame in read_batches():
        frame = frame.astype(dtypes)
        rows += len(frame)
        memory += int(frame.memory_usage(index=False, deep=True).sum())

        records = (
            tuple(map(_store, row))
            for row in frame.itertuples(index=False, name=None)
        )
        db.executemany(insert, records)

    # The reference DataFrame uses one continuous index after concatenation.
    memory += pd.RangeIndex(rows).memory_usage()
    return rows, memory


def _numeric_stats(db, field, n):
    """Compute global numeric statistics and IQR outliers for one column."""
    if n == 0:
        return None, None

    ordered_query = (
        f"SELECT {field} FROM observations "
        f"WHERE {field} IS NOT NULL ORDER BY {field}"
    )

    # pandas uses linear interpolation between adjacent order statistics.
    positions = [(n - 1) * p for p in (0.25, 0.50, 0.75)]
    wanted_ranks = {0, n - 1}

    for position in positions:
        wanted_ranks.update((math.floor(position), math.ceil(position)))

    ranks = {}
    for rank, (value,) in enumerate(db.execute(ordered_query)):
        if rank in wanted_ranks:
            ranks[rank] = value

    def quantile(position):
        low = ranks[math.floor(position)]
        high = ranks[math.ceil(position)]

        if position.is_integer():
            return float(low)

        return float(np.quantile([low, high], position % 1))

    q1, median, q3 = map(quantile, positions)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_query = (
        f"SELECT COUNT(*) FROM observations "
        f"WHERE {field} < ? OR {field} > ?"
    )
    outside = db.execute(outlier_query, (lower, upper)).fetchone()[0]

    def values():
        query = (
            f"SELECT {field} FROM observations "
            f"WHERE {field} IS NOT NULL"
        )
        return (value for (value,) in db.execute(query))

    minimum = ranks[0]
    maximum = ranks[n - 1]
    finite = math.isfinite(minimum) and math.isfinite(maximum)

    mean = statistics.mean(values()) if finite else None
    std = statistics.stdev(values()) if finite and n > 1 else None

    stats = {
        "mean": mean,
        "std": std,
        "min": minimum,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": maximum,
    }

    outliers = {
        "method": "IQR",
        "lower_bound": _safe_number(lower),
        "upper_bound": _safe_number(upper),
        "outlier_count": outside,
        "outlier_percentage": round(100 * outside / n, 2),
    }

    return {key: _safe_number(value) for key, value in stats.items()}, outliers


def _pearson(db, left, right, batch_size):
    """Calculate Pearson correlation using aligned, finite observation pairs."""
    query = (
        f"SELECT {left}, {right} FROM observations "
        f"WHERE {left} IS NOT NULL AND {right} IS NOT NULL"
    )
    cursor = db.execute(query)

    count = 0
    mean = np.zeros(2)
    sums = np.zeros((2, 2))

    while pairs := cursor.fetchmany(batch_size):
        block = np.asarray(pairs, dtype=float)
        block = block[np.isfinite(block).all(axis=1)]
        size = len(block)

        if size == 0:
            continue

        if count == 0:
            origin = block[0].copy()

        # Center around a fixed origin for improved numerical stability.
        block -= origin
        local_mean = block.mean(axis=0)
        centered = block - local_mean

        # Combine this block's centered sums with all previous blocks.
        delta = local_mean - mean
        new_count = count + size

        sums += centered.T @ centered
        sums += np.outer(delta, delta) * (count * size / new_count)
        mean += delta * (size / new_count)
        count = new_count

    if count < 2 or sums[0, 0] <= 0 or sums[1, 1] <= 0:
        return None

    correlation = sums[0, 1] / math.sqrt(sums[0, 0]) / math.sqrt(sums[1, 1])
    return _safe_number(correlation)


def _report(db, schema, fields, rows, memory, options, batch_size):
    """Build the final report in the same layout as the existing profiler."""
    # The empty schema lets us reuse the reference report structure and
    # its target/identifier naming rules without loading all rows into RAM.
    report = profile_dataset(schema, **options)

    grouping = ", ".join(fields)
    distinct_rows_query = (
        "SELECT COUNT(*) FROM "
        f"(SELECT 1 FROM observations GROUP BY {grouping})"
    )
    distinct_rows = db.execute(distinct_rows_query).fetchone()[0]

    report["dataset"].update(
        rows=rows,
        duplicate_rows=rows - distinct_rows,
        memory_usage_bytes=memory,
        memory_usage_mb=round(memory / 1024**2, 2),
    )

    numeric = []

    for field, column in zip(fields, report["columns"]):
        count_query = (
            f"SELECT COUNT({field}), COUNT(DISTINCT {field}) "
            "FROM observations"
        )
        non_null, unique = db.execute(count_query).fetchone()
        missing = rows - non_null

        column.update(
            non_null_count=non_null,
            missing_count=missing,
            unique_count=unique,
            missing_percentage=round(missing / rows * 100, 2) if rows else 0.0,
            is_constant=unique <= 1,
        )

        if "numeric_statistics" in column:
            numeric.append((field, column["name"]))
            stats, outliers = _numeric_stats(db, field, non_null)
            column["numeric_statistics"] = stats
            column["outlier_analysis"] = outliers
            continue

        # For equal counts, list the value seen first in the source first.
        top_query = (
            f"SELECT {field}, COUNT(*) AS freq FROM observations "
            f"WHERE {field} IS NOT NULL GROUP BY {field} "
            "ORDER BY freq DESC, MIN(rowid) LIMIT 10"
        )
        frequencies = [
            {"value": _restore(value), "count": count}
            for value, count in db.execute(top_query)
        ]

        column["categorical_statistics"] = {
            "cardinality": unique,
            "top_frequencies": frequencies,
        }

        if non_null:
            majority = frequencies[0]["count"] / non_null
            column["class_balance"] = {
                "class_count": unique,
                "majority_class_proportion": round(majority, 4),
                "potentially_imbalanced": (
                    unique > 1 and majority >= options["imbalance_threshold"]
                ),
            }

    selected = numeric[:options["max_correlation_columns"]]

    if options["include_correlations"] and len(selected) >= 2:
        matrix = [[None] * len(selected) for _ in selected]

        for i, (left, _) in enumerate(selected):
            for j in range(i + 1):
                right = selected[j][0]
                result = _pearson(db, left, right, batch_size)
                matrix[i][j] = result
                matrix[j][i] = result

        report["correlations"] = {
            "method": "pearson",
            "columns": [name for _, name in selected],
            "matrix": matrix,
            "truncated": len(selected) < len(numeric),
        }

    return report


def profile_loaded_dataset(
    storage_dir,
    dataset_id,
    *,
    batch_size=50_000,
    work_dir=None,
    include_correlations=True,
    max_correlation_columns=50,
    imbalance_threshold=0.8,
):
    """Profile a registered dataset and return a single dataset-wide report.

    storage_dir and dataset_id are the same values used by iter_dataset().
    work_dir optionally selects the parent folder for the temporary database.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if max_correlation_columns < 1:
        raise ValueError("max_correlation_columns must be at least 1")

    if not 0 < imbalance_threshold <= 1:
        raise ValueError("imbalance_threshold must be in (0, 1]")

    def read_batches():
        return iter_dataset(storage_dir, dataset_id, batch_size=batch_size)

    schema = _schema(read_batches)

    # Only internally generated names go into SQL. The user's original
    # column names remain in the pandas schema and in the returned report.
    fields = [f"c{i}" for i in range(len(schema.columns))]

    options = {
        "include_correlations": include_correlations,
        "max_correlation_columns": max_correlation_columns,
        "imbalance_threshold": imbalance_threshold,
    }

    with TemporaryDirectory(dir=work_dir) as folder:
        database_path = Path(folder) / "profile.sqlite"

        # Close SQLite before removing the temporary folder on Windows.
        with closing(sqlite3.connect(database_path)) as db:
            db.execute("PRAGMA temp_store=FILE")
            rows, memory = _stage(db, read_batches, schema, fields)
            return _report(db, schema, fields, rows, memory, options, batch_size)