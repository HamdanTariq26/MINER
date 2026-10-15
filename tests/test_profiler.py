#   Copyright 2026 Hamdan Tariq, Nawal Fatima, Toaseen Arif
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pandas as pd

# Pytest for integration testing
import pytest

from src.data_mining.profiler import profile_dataset


# Note function beginning by 'test' are automatically discoverded by pytest and report are
# generated from them. ( :) general functionality of pytest)

# Check the overview of dataset i.e column and row counts etc
def test_dataset_overview():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Quetta", "Karachi", "Lahore"],
    })

    report = profile_dataset(df)

    assert report["dataset"]["rows"] == 3
    assert report["dataset"]["columns"] == 2
    assert report["dataset"]["memory_usage_bytes"] > 0


def test_missing_values():
    df = pd.DataFrame({
        "score": [10.0, None, 30.0, None],
    })

    report = profile_dataset(df)
    column = report["columns"][0]

    # debugging aid specifying condition to test
    assert column["missing_count"] == 2
    assert column["non_null_count"] == 2
    assert column["missing_percentage"] == 50.0


def test_numeric_statistics():
    df = pd.DataFrame({
        "score": [10, 20, 30, 40, 50],
    })

    report = profile_dataset(df)
    stats = report["columns"][0]["numeric_statistics"]

    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["median"] == 30
    assert stats["mean"] == 30


def test_categorical_frequencies():
    df = pd.DataFrame({
        "city": ["Quetta", "Karachi", "Quetta", "Lahore", "Quetta"],
    })

    report = profile_dataset(df)
    stats = report["columns"][0]["categorical_statistics"]

    assert stats["cardinality"] == 3
    assert stats["top_frequencies"][0]["value"] == "Quetta"
    assert stats["top_frequencies"][0]["count"] == 3


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Quetta", "Karachi", "Lahore"],
    })
    original = df.copy(deep=True)

    profile_dataset(df)

    pd.testing.assert_frame_equal(df, original)


def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        profile_dataset([1, 2, 3])


def test_invalid_imbalance_threshold_raises_value_error():
    df = pd.DataFrame({"category": ["A", "B"]})

    with pytest.raises(ValueError, match="imbalance_threshold"):
        profile_dataset(df, imbalance_threshold=0)


def test_correlations_are_calculated():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8],
    })

    report = profile_dataset(df)

    correlations = report["correlations"]

    assert correlations["method"] == "pearson"
    assert correlations["columns"] == ["x", "y"]
    assert correlations["matrix"][0][1] == pytest.approx(1.0)


def test_correlations_can_be_disabled():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [3, 2, 1],
    })

    report = profile_dataset(df, include_correlations=False)

    assert report["correlations"] is None


def test_outlier_analysis():
    df = pd.DataFrame({
        "score": [10, 11, 12, 13, 100],
    })

    report = profile_dataset(df)
    outliers = report["columns"][0]["outlier_analysis"]

    assert outliers["method"] == "IQR"
    assert outliers["outlier_count"] == 1


def test_class_imbalance():
    df = pd.DataFrame({
        "category": ["A"] * 9 + ["B"],
    })

    report = profile_dataset(df)
    balance = report["columns"][0]["class_balance"]

    assert balance["class_count"] == 2
    assert balance["majority_class_proportion"] == pytest.approx(0.9)
    assert balance["potentially_imbalanced"] is True


def test_target_and_identifier_candidates():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "outcome": ["yes", "no", "yes"],
        "age": [25, 30, 35],
    })

    report = profile_dataset(df)

    target_names = {
        item["column"] for item in report["target_candidates"]
    }
    identifier_names = {
        item["column"] for item in report["identifier_candidates"]
    }

    assert "outcome" in target_names
    assert "customer_id" in identifier_names
