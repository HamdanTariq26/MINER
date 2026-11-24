import pytest

from src.data_mining.loader import load_dataset
from src.data_mining.profiler import profile_loaded_dataset


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_mixed_data_across_batches(tmp_path, batch_size):
    source = tmp_path / "customers.csv"
    source.write_text(
        "CustomerID,City,Income,Empty\n"
        "001,Lahore,100,\n"
        "002,Karachi,,\n"
        "003,Lahore,200,\n"
        "004,Karachi,300,\n"
        "001,Lahore,100,\n",
        encoding="utf-8",
    )

    storage = tmp_path / "datasets"
    dataset_id = load_dataset(source, storage)["dataset_id"]

    reference = profile_loaded_dataset(
    storage, dataset_id, engine="pandas", batch_size=batch_size
    )
    streamed = profile_loaded_dataset(
    storage, dataset_id, engine="streaming", batch_size=batch_size
    )

    for report in (reference, streamed):
        columns = {col["name"]: col for col in report["columns"]}

        assert report["dataset"]["rows"] == 5
        assert report["dataset"]["duplicate_rows"] == 1
        assert columns["CustomerID"]["unique_count"] == 4

        city = columns["City"]
        assert city["categorical_statistics"]["top_frequencies"] == [
            {"value": "Lahore", "count": 3},
            {"value": "Karachi", "count": 2},
        ]
        assert city["class_balance"]["majority_class_proportion"] == 0.6

        income = columns["Income"]
        assert income["missing_count"] == 1
        assert income["numeric_statistics"]["mean"] == 175
        assert income["numeric_statistics"]["median"] == 150

        assert columns["Empty"]["missing_count"] == 5
        assert columns["Empty"]["unique_count"] == 0
        assert columns["Empty"]["class_balance"] is None