import pandas as pd
import pytest
from openpyxl import Workbook

from src.data_mining.loader import iter_dataset, load_dataset
from src.data_mining.profiler import profile_dataset


def test_csv_snapshot_and_schema(tmp_path):
    source = tmp_path / "customers.csv"
    source.write_text('id;amount;city\n001;2;"A;B"\n002;3.5;NA\n003;;\n')
    root = tmp_path / "datasets"
    loaded = load_dataset(source, root)
    source.unlink()
    batches = list(iter_dataset(root, loaded["dataset_id"], batch_size=2))
    assert [len(frame) for frame in batches] == [2, 1]
    assert str(batches[1]["amount"].dtype) == "Float64"
    df = pd.concat(batches, ignore_index=True)
    assert list(df.id) == ["001", "002", "003"]
    assert list(df.city[:2]) == ["A;B", "NA"]
    assert df.amount.isna().sum() == 1
    assert profile_dataset(df)["dataset"]["rows"] == 3


@pytest.mark.parametrize("contents", ["a,b\n1,2\n3\n", "a,a\n1,2\n", "a,b\n1,\"bad"])
def test_failed_import_leaves_no_dataset(tmp_path, contents):
    import csv

    source = tmp_path / "broken.csv"
    source.write_text(contents)
    root = tmp_path / "store"
    with pytest.raises((ValueError, csv.Error)):
        load_dataset(source, root)
    assert list(root.iterdir()) == []


def test_workbook_sheet_selection_and_header(tmp_path):
    book = Workbook()
    sheet = book.create_sheet("Customers")
    sheet.append(["Report"])
    sheet.append(["name", "age"])
    sheet.append(["Ali", 22])
    source = tmp_path / "book.xlsx"
    book.save(source)
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="Customers"):
        load_dataset(source, root)
    result = load_dataset(source, root, sheet="Customers", header_row=2)
    frame = next(iter_dataset(root, result["dataset_id"]))
    assert frame.to_dict("records") == [{"name": "Ali", "age": 22}]


def test_header_only_csv(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_text("amount\n")
    loaded = load_dataset(source, tmp_path, delimiter=",")
    frames = list(iter_dataset(tmp_path, loaded["dataset_id"]))
    assert len(frames) == 1 and frames[0].empty
    assert list(frames[0]) == ["amount"]
