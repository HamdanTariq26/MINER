import numpy as np
import pandas as pd
import pytest

from src.data_mining.loader import load_dataset
from src.data_mining.preprocessing import (
    fit_preprocessor,
    iter_preprocessed_partition,
    transform_batch,
)
from src.data_mining.splitting import iter_partition, split_dataset


@pytest.mark.parametrize("batch_size", [1, 4, 50])
def test_fit_on_train_and_reuse_across_batches(tmp_path, batch_size):
    source = tmp_path / "customers.csv"
    frame = pd.DataFrame({
        "Income": [None if i % 3 == 0 else i * 100 for i in range(30)],
        "Class": [i % 2 for i in range(30)],
    })
    frame.to_csv(source, index=False)

    storage = tmp_path / "datasets"
    dataset_id = load_dataset(source, storage)["dataset_id"]
    split = split_dataset(
        storage,
        dataset_id,
        target="Class",
        strategy="stratified",
        fractions={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=42,
    )
    split_id = split["split_id"]

    def raw_partition(name):
        return pd.concat(
            iter_partition(storage, dataset_id, split_id, name, batch_size=7),
            ignore_index=True,
        )

    train = pd.to_numeric(raw_partition("train")["Income"]).astype(float)
    assert 0 < train.isna().sum() < len(train)

    expected_mean = train.mean()
    expected_scale = train.fillna(expected_mean).std(ddof=0)

    fitted = fit_preprocessor(
        storage,
        dataset_id,
        split_id,
        target="Class",
        features=["Income"],
        steps=[
            {"operation": "impute_mean", "column": "Income"},
            {"operation": "standard_scale", "column": "Income"},
        ],
        batch_size=batch_size,
    )

    imputer, scaler = fitted["steps"]
    assert imputer["fill_value"] == pytest.approx(expected_mean)
    assert scaler["mean"] == pytest.approx(expected_mean)
    assert scaler["scale"] == pytest.approx(expected_scale)
    assert scaler["observed_count"] == len(train)

    for partition in ("train", "validation", "test"):
        raw = raw_partition(partition)
        expected = (
            pd.to_numeric(raw["Income"]).astype(float).fillna(expected_mean)
            - expected_mean
        ) / expected_scale

        batches = list(iter_preprocessed_partition(
            storage,
            dataset_id,
            split_id,
            fitted["preprocessor_id"],
            partition,
            batch_size=batch_size,
        ))
        actual_x = pd.concat([x for x, _ in batches], ignore_index=True)
        actual_y = pd.concat([y for _, y in batches], ignore_index=True)

        assert list(actual_x.columns) == ["Income"]
        np.testing.assert_allclose(actual_x["Income"], expected, atol=1e-10)
        assert actual_y.tolist() == raw["Class"].tolist()

    # Simulate unseen inputs arriving together in one batch. The extreme
    # value must have no influence on the value used to fill its neighbor.
    unseen = pd.DataFrame({"Income": [None, 1_000_000]})
    actual = transform_batch(unseen, fitted)
    np.testing.assert_allclose(
        actual["Income"],
        [0.0, (1_000_000 - expected_mean) / expected_scale],
        atol=1e-10,
    )

@pytest.mark.parametrize("batch_size", [1, 4, 50])
def test_drop_columns_and_global_iqr_filter(tmp_path, monkeypatch, batch_size):
    import src.data_mining.preprocessing as prep

    # Row positions in the label make feature/label misalignment detectable.
    train = pd.DataFrame({
        "CustomerID": list("abcdefgh"),
        "Income": [1, 1, 2, 100, 2, 2, 3, 4],
        "Label": list(range(8)),
    })
    validation = pd.DataFrame({
        "CustomerID": ["z"],
        "Income": [10_000],
        "Label": [99],
    })

    def batches(storage_dir, dataset_id, split_id, partition, *, batch_size):
        source = {"train": train, "validation": validation}[partition]
        for start in range(0, len(source), batch_size):
            yield source.iloc[start:start + batch_size].copy()

    monkeypatch.setattr(prep, "iter_partition", batches)

    fitted = prep.fit_preprocessor(
        tmp_path,
        "dataset",
        "split",
        target="Label",
        features=["CustomerID", "Income"],
        steps=[
            {"operation": "drop_columns", "columns": ["CustomerID"]},
            {"operation": "filter_iqr", "column": "Income"},
            {"operation": "standard_scale", "column": "Income"},
        ],
        batch_size=batch_size,
    )

    _, outliers, scaler = fitted["steps"]
    assert fitted["output_features"] == ["Income"]
    assert fitted["training_rows_before"] == 8
    assert fitted["training_rows_after"] == 7
    assert outliers["q1"] == pytest.approx(1.75)
    assert outliers["q3"] == pytest.approx(3.25)
    assert outliers["lower_bound"] == pytest.approx(-0.5)
    assert outliers["upper_bound"] == pytest.approx(5.5)
    assert outliers["training_rows_removed"] == 1

    surviving = train.loc[train["Income"] != 100]
    assert scaler["mean"] == pytest.approx(surviving["Income"].mean())
    assert scaler["scale"] == pytest.approx(surviving["Income"].std(ddof=0))

    def collect(partition):
        parts = list(prep.iter_preprocessed_partition(
            tmp_path, "dataset", "split", fitted["preprocessor_id"],
            partition, batch_size=batch_size,
        ))
        return pd.concat([x for x, _ in parts]), pd.concat([y for _, y in parts])

    x_train, y_train = collect("train")
    assert list(x_train.columns) == ["Income"]
    assert y_train.tolist() == surviving["Label"].tolist()
    np.testing.assert_allclose(
        x_train["Income"],
        (surviving["Income"] - scaler["mean"]) / scaler["scale"],
    )

    x_val, y_val = collect("validation")
    assert y_val.tolist() == [99]
    expected = (10_000 - scaler["mean"]) / scaler["scale"]
    assert x_val["Income"].iloc[0] == pytest.approx(expected)

    prediction = prep.transform_batch(pd.DataFrame({"Income": [10_000]}), fitted)
    assert prediction["Income"].iloc[0] == pytest.approx(expected)