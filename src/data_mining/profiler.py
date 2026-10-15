"""Dataset profiling for the Intelligent Data Mining Assistant."""

import math

import pandas as pd

# Possible targets based of feature name:-> only suggestive
TARGET_NAME_HINTS = {
    "target", "label", "class", "outcome", "result",
    "prediction", "response", "diagnosis",
}

# Possible identifiers based on  features name :-> only suggestive
IDENTIFIER_NAME_HINTS = {
    "id", "uuid", "guid", "identifier", "record_id",
}


def profile_dataset(
    df: pd.DataFrame,
    *,
    # Calculation of correlations are expensive
    include_correlations: bool = True,
    max_correlation_columns: int = 50,

    imbalance_threshold: float = 0.8,
) -> dict:
    """Profile a pandas DataFrame without modifying the input."""
    # Checks type, only allows strictly pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if max_correlation_columns < 1:
        raise ValueError("max_correlation_columns must be at least 1")

    # Valid Imbalance treshold 
    if not 0 < imbalance_threshold <= 1:
        raise ValueError("imbalance_threshold must be in (0, 1]")

    row_count, column_count = df.shape
    memory_usage_bytes = int(
        df.memory_usage(index=True, deep=True).sum()
    )

    column_profiles = []
    numeric_columns = []
    target_candidates = []
    identifier_candidates = []

    for column in df.columns:
        # Reason to extract each is for calculation of each column, it makes the whole 
        # architecture scalable, i.e no matter how many columns we might have.
        # if we called function on whole dataset we need to write complex statements to convert it in
        # json safe and safe data.
        series = df[column]  # full series is never returned in data profiler output.
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique())
        column_name = str(column)

        # normalizing feature names
        normalized_name = column_name.lower().strip()

        profile = {
            "name": column_name,
            "dtype": str(series.dtype),
            "non_null_count": int(row_count - missing_count),
            "missing_count": missing_count,
            "missing_percentage": (
                round(missing_count / row_count * 100, 2)
                if row_count else 0.0
            ),
            "unique_count": unique_count,
            "is_constant": unique_count <= 1,
        }

        # For numeric values
        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric_columns.append(column)
            numeric = series.dropna()

            if not numeric.empty:
                # Detects number of outliers
                q1 = numeric.quantile(0.25)
                median = numeric.median()
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                outlier_count = int(
                    ((numeric < lower) | (numeric > upper)).sum()
                )

                # Calculate statistics
                profile["numeric_statistics"] = {
                    "mean": _safe_number(numeric.mean()),
                    "std": _safe_number(numeric.std()),
                    "min": _safe_number(numeric.min()),
                    "q1": _safe_number(q1),
                    "median": _safe_number(median),
                    "q3": _safe_number(q3),
                    "max": _safe_number(numeric.max()),
                }
                # Outliers analysis
                profile["outlier_analysis"] = {
                    "method": "IQR",
                    "lower_bound": _safe_number(lower),
                    "upper_bound": _safe_number(upper),
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(
                        outlier_count / len(numeric) * 100, 2
                    ),
                }
            else:
                # if empty
                profile["numeric_statistics"] = None
                profile["outlier_analysis"] = None
        # If categorical
        else:
            frequencies = series.value_counts(dropna=True).head(10)  # return top 10 frequencies
            profile["categorical_statistics"] = {
                "cardinality": unique_count,  # gives cardinality of each column
                "top_frequencies": [
                    {
                        "value": _safe_value(value),
                        "count": int(count),
                    }
                    for value, count in frequencies.items()
                ],
            }

            non_null_count = row_count - missing_count

            if non_null_count > 0:

                # calculates relative frequency class distribution, if any class have a treshold greater or equal is considered as imbalance
                proportions = series.value_counts(
                    normalize=True, dropna=True
                )

                # returns only majority proption class
                majority_proportion = float(proportions.iloc[0])

                profile["class_balance"] = {
                    "class_count": unique_count,
                    "majority_class_proportion": round(
                        majority_proportion, 4
                    ),
                    "potentially_imbalanced": (
                        unique_count > 1
                        and majority_proportion >= imbalance_threshold
                    ),
                }
            else:
                profile["class_balance"] = None

        # Extracts prefixes and suffices
        name_parts = set(normalized_name.replace("-", "_").split("_"))

        # Suggest potentiol targets and identifiers
        if (
            name_parts & TARGET_NAME_HINTS
            or normalized_name in TARGET_NAME_HINTS
        ):
            target_candidates.append({
                "column": column_name,
                "reasons": ["Column name suggests a target or outcome"],
                "confidence": "weak",
            })

        if (
            normalized_name in IDENTIFIER_NAME_HINTS
            or normalized_name.endswith("_id")
            or normalized_name.endswith("_uuid")
        ):
            identifier_candidates.append({
                "column": column_name,
                "reasons": ["Column name suggests an identifier"],
            })

        column_profiles.append(profile)  # build full column dataset profile

    # Calculate correlations
    correlations = None
    correlation_columns = numeric_columns[:max_correlation_columns]  # contains numeric column names only

    # if correlations is set true and at least two columns are given calculate
    if include_correlations and len(correlation_columns) >= 2:
        matrix = df[correlation_columns].corr(method="pearson")

        correlations = {
            "method": "pearson",
            "columns": [str(column) for column in correlation_columns],  # columns to calculate pearson correlation
            # Gurantes that there is no numpy specific datatype
            "matrix": [
                [_safe_number(value) for value in row]
                for row in matrix.to_numpy().tolist()  # it converts 2D pandas dataframe in to standard python list of list. That can be easily converted into json and sent to frontened
            ],
            "truncated": len(numeric_columns) > max_correlation_columns,  # tells if correlation of every column is computed or not
        }

    # Returns a structured output
    return {
        "dataset": {
            "rows": int(row_count),
            "columns": int(column_count),
            "memory_usage_bytes": memory_usage_bytes,
            "memory_usage_mb": round(
                memory_usage_bytes / (1024 ** 2), 2
            ),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "columns": column_profiles,
        "correlations": correlations,
        "target_candidates": target_candidates,
        "identifier_candidates": identifier_candidates,
    }


# Converts values in json safe python values
def _safe_number(value):
    """Convert numeric results to JSON-safe Python values."""
    if pd.isna(value):
        return None

    # hasattr check if an attribute has a specific attribute
    # here it checks for an attribute name item
    result = value.item() if hasattr(value, "item") else value

    # .item() function converts numpy values into python values

    # if that item value is float it checks if it is an infinite number or -inif, it true it returns
    # false
    if isinstance(result, float):
        return result if math.isfinite(result) else None

    return result


# Converts values in json safe python values - light version
def _safe_value(value):
    """Convert common pandas and NumPy scalars to Python values."""
    if pd.isna(value):
        return None

    return value.item() if hasattr(value, "item") else value