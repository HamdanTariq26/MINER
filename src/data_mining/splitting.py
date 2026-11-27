"""Persist reusable train, validation and test assignments for one dataset."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from .loader import iter_dataset


PARTITIONS = ("train", "validation", "test")


def _allocate(total, weights):
    """Round proportional quotas while keeping the total exact."""
    quotas = total * (weights / weights.sum())
    counts = np.floor(quotas).astype(np.int64)

    remainder = total - int(counts.sum())
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[:remainder]] += 1

    return counts


def split_dataset(storage_dir, dataset_id, *, target, strategy, fractions, seed):
    """Create a row-position split for a registered dataset snapshot."""
    dataset_id = str(UUID(dataset_id))
    root = Path(storage_dir)
    manifest_path = root / dataset_id / "manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8"))

    if target not in source["columns"]:
        raise ValueError(f"Unknown target column: {target}")

    if strategy not in ("random", "stratified"):
        raise ValueError("Choose random or stratified")

    if not isinstance(fractions, dict) or set(fractions) != set(PARTITIONS):
        raise ValueError(
            "Supply train, validation and test fractions; use 0 for test if unused"
        )

    if any(type(fractions[name]) not in (float, int) for name in PARTITIONS):
        raise ValueError("Fractions must be numbers")

    weights = np.array([fractions[name] for name in PARTITIONS], dtype=float)

    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or min(weights[:2]) <= 0
        or abs(weights.sum() - 1) > 1e-9
    ):
        raise ValueError(
            "Fractions must sum to 1 with positive train and validation fractions"
        )

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    n = source["row_count"]
    sizes = _allocate(n, weights)
    active = np.flatnonzero(weights > 0)

    if (sizes[active] == 0).any():
        raise ValueError("Too few rows for the requested fractions")

    labels = []
    seen = 0

    for frame in iter_dataset(root, dataset_id):
        series = frame[target]

        if series.isna().any():
            raise ValueError(
                "Target contains missing labels; resolve them before splitting"
            )

        if strategy == "stratified":
            labels.append(series.copy())

        seen += len(frame)

    if seen != n:
        raise ValueError("Snapshot row count differs from its manifest")

    rng = np.random.default_rng(seed)
    assignment = np.empty(n, dtype=np.uint8)

    if strategy == "random":
        rows = rng.permutation(n)
        start = 0

        for partition in active:
            stop = start + sizes[partition]
            assignment[rows[start:stop]] = partition
            start = stop

    else:
        target_values = pd.concat(labels, ignore_index=True)
        codes, classes = pd.factorize(target_values, sort=False)
        counts = np.bincount(codes)

        rare = int(np.argmin(counts))
        if counts[rare] < len(active):
            raise ValueError(
                f"Class {classes[rare]!r} has only {counts[rare]} rows "
                f"for {len(active)} partitions"
            )

        if min(sizes[active]) < len(classes):
            raise ValueError("A partition is too small to include every class")

        # Reserve one example of each class for every requested partition.
        slots = np.zeros((len(classes), len(PARTITIONS)), dtype=np.int64)
        slots[:, active] = 1
        remaining = counts - len(active)

        # Allocate the remaining examples while meeting exact partition sizes.
        for partition in active[:-1]:
            needed = int(sizes[partition]) - len(classes)

            if needed:
                extra = _allocate(needed, remaining)
                slots[:, partition] += extra
                remaining -= extra

        slots[:, active[-1]] += remaining

        # Shuffle within each class before assigning its rows.
        rows = np.argsort(codes, kind="stable")
        start = 0

        for class_index, count in enumerate(counts):
            group = rows[start:start + count]
            rng.shuffle(group)
            assignment[group] = np.repeat(active, slots[class_index, active])
            start += count

    split_id = str(uuid4())
    result = {
        "status": "ok",
        "tool": "split_dataset",
        "split_id": split_id,
        "dataset_id": dataset_id,
        "dataset_version": source["version"],
        "target": target,
        "strategy": strategy,
        "fractions": fractions,
        "seed": seed,
        "rows": n,
        "counts": dict(zip(PARTITIONS, sizes.tolist())),
    }

    directory = root / dataset_id / "splits"
    directory.mkdir(exist_ok=True)

    # Publish the split only after both files have been written.
    with TemporaryDirectory(dir=directory) as temp:
        staged = Path(temp) / split_id
        staged.mkdir()
        np.save(staged / "assignments.npy", assignment, allow_pickle=False)
        (staged / "split.json").write_text(json.dumps(result), encoding="utf-8")
        staged.rename(directory / split_id)

    return result


def iter_partition(storage_dir, dataset_id, split_id, partition, *, batch_size=50000):
    """Read a saved partition in source order using any loader batch size."""
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")

    dataset_id = str(UUID(dataset_id))
    split_id = str(UUID(split_id))
    path = Path(storage_dir) / dataset_id / "splits" / split_id / "assignments.npy"

    assignment = np.load(path, mmap_mode="r", allow_pickle=False)
    code = PARTITIONS.index(partition)
    start = 0

    for frame in iter_dataset(storage_dir, dataset_id, batch_size=batch_size):
        stop = start + len(frame)
        selected = assignment[start:stop] == code

        if selected.any():
            frame.index = pd.RangeIndex(start, stop)
            yield frame.loc[selected]

        start = stop

    if start != len(assignment):
        raise ValueError("Split and dataset row counts differ")