"""Fit ordered preprocessing steps on training rows and reuse them on batches."""

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd

from .splitting import iter_partition


_STEP_FIELDS = {
    "drop_columns": {"operation", "columns"},
    "impute_mean": {"operation", "column"},
    "impute_median": {"operation", "column"},
    "impute_mode": {"operation", "column"},
    "impute_constant": {"operation", "column", "value"},
    "clip_iqr": {"operation", "column", "factor"},
    "filter_iqr": {"operation", "column", "factor"},
    "standard_scale": {"operation", "column"},
    "minmax_scale": {"operation", "column"},
}

_ORDERED_NUMERIC = {"impute_median", "clip_iqr", "filter_iqr"}


def _validate_columns(names, label):
    if not isinstance(names, list) or not names:
        raise ValueError(f"{label} must be a non-empty list.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain non-empty column names.")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate column names.")


def _json_scalar(value):
    """Keep strings, booleans and numbers distinct in saved artifacts."""
    if isinstance(value, np.generic):
        value = value.item()

    if type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value

    raise ValueError(f"Expected a finite number, string or boolean; got {value!r}.")


def _validate_plan(features, target, steps):
    _validate_columns(features, "features")

    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty column name.")
    if target in features:
        raise ValueError(f"Target {target!r} cannot be a feature.")
    if not isinstance(steps, list):
        raise ValueError("steps must be an ordered list.")

    active = list(features)
    normalized = []

    for index, raw in enumerate(steps, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Step {index} must be an object.")

        step = dict(raw)
        op = step.get("operation")
        if not isinstance(op, str) or op not in _STEP_FIELDS:
            raise ValueError(f"Step {index}: unsupported operation {op!r}.")

        if op in {"clip_iqr", "filter_iqr"}:
            step.setdefault("factor", 1.5)

        if set(step) != _STEP_FIELDS[op]:
            raise ValueError(
                f"Step {index} ({op}) expects fields {sorted(_STEP_FIELDS[op])}."
            )

        names = step["columns"] if op == "drop_columns" else [step["column"]]
        _validate_columns(names, f"Step {index} columns")
        unavailable = [name for name in names if name not in active]
        if unavailable:
            raise ValueError(
                f"Step {index} ({op}) references unavailable features: {unavailable}."
            )

        if op == "drop_columns":
            active = [name for name in active if name not in names]
            if not active:
                raise ValueError("The plan would drop every feature.")
        elif op in {"clip_iqr", "filter_iqr"}:
            factor = step["factor"]
            if type(factor) not in (int, float) or not math.isfinite(factor):
                raise ValueError("IQR factor must be a finite number.")
            if factor <= 0:
                raise ValueError("IQR factor must be greater than zero.")
        elif op == "impute_constant":
            step["value"] = _json_scalar(step["value"])

        normalized.append(step)

    return normalized, active


def _numeric(series):
    """Convert numeric inputs while keeping missing values distinguishable."""
    try:
        numeric = pd.to_numeric(series, errors="raise")
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Column {series.name!r} requires numeric values.") from exc

    if np.isinf(values).any():
        raise ValueError(f"Column {series.name!r} contains infinity.")
    return values


def _apply(batch, features, fitted_steps, *, training):
    """Return transformed features and original row positions that survived."""
    missing = [name for name in features if name not in batch.columns]
    if missing:
        raise ValueError(f"Input batch is missing features: {missing}")

    frame = batch.loc[:, features].copy()
    positions = np.arange(len(frame))

    for step in fitted_steps:
        op = step["operation"]

        if op == "drop_columns":
            frame = frame.drop(columns=step["columns"])
            continue

        if op == "filter_iqr" and not training:
            continue

        column = step["column"]

        if op in {"impute_constant", "impute_mode"}:
            series = frame[column]
            if series.isna().any():
                try:
                    frame[column] = series.fillna(step["fill_value"])
                except (TypeError, ValueError):
                    # Nullable and categorical pandas dtypes can reject a new
                    # category or a non-integral fill value. Preserve the scalar.
                    frame[column] = series.astype(object).where(
                        series.notna(), step["fill_value"]
                    )
            continue

        values = _numeric(frame[column])

        if op in {"impute_mean", "impute_median"}:
            values[np.isnan(values)] = step["fill_value"]
        elif op == "standard_scale":
            values = (values - step["mean"]) / step["scale"]
        elif op == "minmax_scale":
            values = (values - step["minimum"]) / step["scale"]
        elif op == "clip_iqr":
            values = np.clip(values, step["lower_bound"], step["upper_bound"])
        elif op == "filter_iqr":
            # An unobserved value has no measured distance beyond either fence.
            outside = (values < step["lower_bound"]) | (values > step["upper_bound"])
            keep = np.flatnonzero(~outside)
            frame = frame.iloc[keep].copy()
            positions = positions[keep]
            continue
        else:
            raise ValueError(f"Unknown saved operation: {op!r}")

        if np.isinf(values).any():
            raise ValueError(f"Numeric overflow in {op} on {column!r}.")
        frame[column] = values

    return frame, positions


def _quantile(db, n, probability):
    """Linear interpolation at the same zero-based rank used by pandas."""
    position = (n - 1) * probability
    lower_rank = math.floor(position)
    fraction = position - lower_rank
    rows = db.execute(
        "SELECT value FROM vals ORDER BY value LIMIT 2 OFFSET ?",
        (lower_rank,),
    ).fetchall()

    low = rows[0][0]
    if fraction == 0:
        return low
    return float(np.quantile([low, rows[1][0]], fraction))


def _fit_ordered(step, series_batches, work_dir):
    """Use temporary disk storage for exact global training quantiles."""
    op = step["operation"]
    column = step["column"]

    with TemporaryDirectory(dir=work_dir) as temporary:
        with closing(sqlite3.connect(Path(temporary) / "quantiles.sqlite")) as db:
            db.execute("CREATE TABLE vals (value REAL NOT NULL)")
            n = 0
            for series in series_batches():
                values = _numeric(series)
                observed = values[~np.isnan(values)]
                db.executemany(
                    "INSERT INTO vals VALUES (?)",
                    ((float(value),) for value in observed),
                )
                n += len(observed)

            if n == 0:
                raise ValueError(f"Cannot fit {op}: {column!r} has no observed values.")

            db.execute("CREATE INDEX vals_order ON vals(value)")
            fitted = {**step, "observed_count": n}

            if op == "impute_median":
                fitted["fill_value"] = _quantile(db, n, 0.5)
                return fitted

            q1 = _quantile(db, n, 0.25)
            q3 = _quantile(db, n, 0.75)
            iqr = q3 - q1
            lower = q1 - step["factor"] * iqr
            upper = q3 + step["factor"] * iqr
            outside = db.execute(
                "SELECT COUNT(*) FROM vals WHERE value < ? OR value > ?",
                (lower, upper),
            ).fetchone()[0]

            fitted.update(
                q1=q1,
                q3=q3,
                iqr=iqr,
                lower_bound=lower,
                upper_bound=upper,
                training_values_outside=outside,
            )
            if op == "filter_iqr":
                fitted["training_rows_removed"] = outside
            return fitted


def _fit_mode(step, series_batches, work_dir):
    """Count across batches on disk. Break ties by first training occurrence."""
    with TemporaryDirectory(dir=work_dir) as temporary:
        with closing(sqlite3.connect(Path(temporary) / "mode.sqlite")) as db:
            db.execute(
                "CREATE TABLE counts "
                "(value TEXT PRIMARY KEY, frequency INTEGER, first_seen INTEGER)"
            )
            observed = 0
            for series in series_batches():
                local = {}
                for value in series.dropna():
                    key = json.dumps(_json_scalar(value), allow_nan=False)
                    if key not in local:
                        local[key] = [0, observed]
                    local[key][0] += 1
                    observed += 1

                db.executemany(
                    "INSERT INTO counts VALUES (?, ?, ?) "
                    "ON CONFLICT(value) DO UPDATE SET "
                    "frequency = counts.frequency + excluded.frequency",
                    ((key, count, first) for key, (count, first) in local.items()),
                )

            row = db.execute(
                "SELECT value FROM counts "
                "ORDER BY frequency DESC, first_seen ASC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError(f"Cannot fit mode: {step['column']!r} is all missing.")
            return {
                **step,
                "fill_value": json.loads(row[0]),
                "observed_count": observed,
            }


def _fit_numeric_summary(step, series_batches):
    op = step["operation"]
    count = 0
    mean = 0.0
    m2 = 0.0
    minimum = math.inf
    maximum = -math.inf

    for series in series_batches():
        values = _numeric(series)
        observed = values[~np.isnan(values)]
        size = len(observed)
        if size == 0:
            continue

        total = count + size
        if op == "minmax_scale":
            minimum = min(minimum, float(observed.min()))
            maximum = max(maximum, float(observed.max()))
        else:
            batch_mean = float(observed.mean())
            delta = batch_mean - mean
            if op == "standard_scale":
                centered = observed - batch_mean
                m2 += float(np.dot(centered, centered))
                m2 += delta * delta * count * size / total
            mean += delta * (size / total)

        count = total

    if count == 0:
        raise ValueError(f"Cannot fit {op}: {step['column']!r} is all missing.")

    fitted = {**step, "observed_count": count}
    if op == "impute_mean":
        fitted["fill_value"] = mean
    elif op == "standard_scale":
        fitted.update(mean=mean, scale=math.sqrt(m2 / count) or 1.0)
    elif op == "minmax_scale":
        fitted.update(minimum=minimum, maximum=maximum, scale=(maximum - minimum) or 1.0)
    return fitted


def fit_preprocessor(
    storage_dir,
    dataset_id,
    split_id,
    *,
    target,
    features,
    steps,
    batch_size=50_000,
    work_dir=None,
):
    """Fit a Qwen-selected plan. Publish an artifact only after a successful fit."""
    plan, output_features = _validate_plan(features, target, steps)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    def training_batches():
        return iter_partition(
            storage_dir, dataset_id, split_id, "train", batch_size=batch_size
        )

    fitted = []

    def series_batches(column):
        for batch in training_batches():
            frame, _ = _apply(batch, features, fitted, training=True)
            yield frame[column]

    for step in plan:
        op = step["operation"]

        if op == "drop_columns":
            fitted_step = step
        elif op == "impute_constant":
            fitted_step = {**step, "fill_value": step["value"]}
        else:
            def read_column():
                return series_batches(step["column"])

            if op in _ORDERED_NUMERIC:
                fitted_step = _fit_ordered(step, read_column, work_dir)
            elif op == "impute_mode":
                fitted_step = _fit_mode(step, read_column, work_dir)
            else:
                fitted_step = _fit_numeric_summary(step, read_column)

        fitted.append(fitted_step)

    # Validate the complete transformation and count surviving rows, including
    # plans with no learned steps or with a filter as the final operation.
    before = 0
    after = 0
    for batch in training_batches():
        if target not in batch.columns:
            raise ValueError(f"Training data is missing target {target!r}.")
        frame, _ = _apply(batch, features, fitted, training=True)
        before += len(batch)
        after += len(frame)

    if after == 0:
        raise ValueError("The training partition has no rows after preprocessing.")

    artifact = {
        "version": 2,
        "preprocessor_id": str(uuid4()),
        "dataset_id": dataset_id,
        "split_id": split_id,
        "fit_partition": "train",
        "target": target,
        "features": list(features),
        "output_features": output_features,
        "training_rows_before": before,
        "training_rows_after": after,
        "steps": fitted,
    }

    payload = json.dumps(artifact, indent=2, allow_nan=False)
    directory = Path(storage_dir) / dataset_id / "splits" / split_id / "preprocessors"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact['preprocessor_id']}.json"
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return {"status": "ok", "tool": "fit_preprocessor", **artifact}


def transform_batch(batch, preprocessor):
    """Transform new prediction rows and preserve their original order."""
    # Dropped columns are unnecessary at prediction time. Every supported
    # transformation is column-local, and training filters are skipped here.
    output_features = preprocessor.get("output_features", preprocessor["features"])
    steps = [
        step for step in preprocessor["steps"]
        if step["operation"] not in {"drop_columns", "filter_iqr"}
        and step["column"] in output_features
    ]
    frame, _ = _apply(batch, output_features, steps, training=False)
    return frame


def iter_preprocessed_partition(
    storage_dir,
    dataset_id,
    split_id,
    preprocessor_id,
    partition,
    *,
    batch_size=50_000,
):
    """Yield aligned feature and label batches from a saved partition."""
    path = (
        Path(storage_dir) / dataset_id / "splits" / split_id
        / "preprocessors" / f"{preprocessor_id}.json"
    )
    preprocessor = json.loads(path.read_text(encoding="utf-8"))
    target = preprocessor["target"]

    for batch in iter_partition(
        storage_dir, dataset_id, split_id, partition, batch_size=batch_size
    ):
        if partition == "train":
            frame, positions = _apply(
                batch, preprocessor["features"], preprocessor["steps"], training=True
            )
            if len(frame):
                yield frame, batch[target].iloc[positions]
        else:
            yield transform_batch(batch, preprocessor), batch[target]