import pandas as pd
import json

#inspect data
def inspect_data(df):
    df.info()
    if df.empty:
        return {}
    return json.loads(df.describe(include="all").to_json())

#handle missing values
def handle_missing_values(df, means, skip_columns=()):
    for column in df.columns:
        if column in skip_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if column in means:
                df[column] = df[column].fillna(means[column])
        else:
            df[column] = df[column].fillna("unknown")
    return df

#remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates()

#fix data types
def fix_data_types(df, numeric_columns=None, skip_columns=()):
    if numeric_columns is not None:
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="raise").astype("Float64")
        return df

    for column in df.columns:
        values = df[column].dropna().astype("string")
        if column in skip_columns or values.empty:
            continue

    # Preserve padded IDs and long integer codes.
        digits = values.str.lstrip("+-")
        padded = digits.str.startswith("0") & digits.str.len().gt(1)
        if (digits.str.isdecimal() & (padded | digits.str.len().gt(15))).any():
            continue
      #convert to numeric
        numbers = pd.to_numeric(df[column], errors="coerce").astype("Float64")
        numbers = numbers.replace([float("inf"), -float("inf")], pd.NA)
        if numbers.notna().sum() == df[column].notna().sum():
            df[column] = numbers
    return df

#clean column names
def clean_column_names(df):
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
    )
    if not df.columns.is_unique:
        raise ValueError("Column names became duplicates after cleaning.")
    return df


#clean text
def clean_text(df, lowercase_columns=()):
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip()
        if column in lowercase_columns:
            df[column] = df[column].str.lower()
    return df.replace("", pd.NA)

#handle outliers
def handle_outliers(df, reference=None, cap=False, skip_columns=()):
    reference = df if reference is None else reference
    counts = {}

    for column in df.select_dtypes(include="number").columns:
        if column in skip_columns:
            continue
        q1, q3 = reference[column].quantile([0.25, 0.75])
        if pd.isna(q1) or pd.isna(q3) or q1 == q3:
            continue

        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts[column] = int(((df[column] < lower) | (df[column] > upper)).sum())
        if cap:
            df[column] = df[column].clip(lower, upper)
    return df, counts


#standardize categories
def standardize_categories(df, replacements):
    for column, mapping in replacements.items():
        df[column] = df[column].str.lower().replace(mapping)
    return df

#fix date formats
def fix_date_formats(df, date_formats):
    for column, date_format in date_formats.items():
        dates = pd.to_datetime(df[column], format=date_format, errors="coerce")
        df[column] = dates.dt.strftime("%Y-%m-%d").astype("string")
    return df

#remove empty rows and columns
def remove_empty_rows_columns(df, keep_columns=None):
    df = df.dropna(how="all")
    if keep_columns is None:
        return df.dropna(axis=1, how="all")
    return df.reindex(columns=keep_columns)


#validate data
def validate_data(df):
    return {
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict()
    }

#generate report
def generate_report(results):
    return json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)


#clean csv
def clean_csv(input_file, output_file="cleaned_data.json",
              chunk_size=10000, sample_size=10000,
              skip_columns=(), lowercase_columns=(),
              category_replacements=None, date_formats=None,
              deduplicate=True, cap_outliers=False):

    input_file, output_file = str(input_file), str(output_file)
    if input_file == output_file:
        raise ValueError("Use a different output filename.")
    if chunk_size < 1 or sample_size < 1:
        raise ValueError("Chunk size and sample size must be positive.")

    category_replacements = category_replacements or {}
    date_formats = date_formats or {}
    skip = set(skip_columns) | set(date_formats) | {
        "id", "target", "label", "name", "phone", "email", "code", "zip",
        "date", "time"
    }
#read chunks
    def read_chunks():
        return pd.read_csv(
            input_file, chunksize=chunk_size, dtype="string",
            keep_default_na=False, skip_blank_lines=False
        )

    def prepare(df):
        df = clean_column_names(df)
        df = clean_text(df, lowercase_columns)
        df = standardize_categories(df, category_replacements)
        return fix_date_formats(df, date_formats)

    # identify columns, calculate global means, keep a small sample.
    totals, counts = {}, {}
    present_columns, not_numeric = set(), set()
    sample = pd.DataFrame()
    columns = []
    rows_read = empty_rows_removed = 0

    with read_chunks() as reader:
        for df in reader:
            rows_read += len(df)
            before = len(df)
            df = prepare(df)
            columns = list(df.columns)
            skip.update(c for c in columns if c.endswith(("_id", "_date", "_time")))
            df = remove_empty_rows_columns(df, keep_columns=columns)
            empty_rows_removed += before - len(df)
            present_columns.update(df.dropna(axis=1, how="all").columns)

            if len(sample) < sample_size:
                part = df.head(sample_size - len(sample))
                sample = pd.concat([sample, part], ignore_index=True)

            typed = fix_data_types(df.copy(), skip_columns=skip)
            for column in columns:
                if typed[column].notna().any():
                    if pd.api.types.is_numeric_dtype(typed[column]):
                        totals[column] = totals.get(column, 0.0) + float(typed[column].sum())
                        counts[column] = counts.get(column, 0) + int(typed[column].count())
                    else:
                        not_numeric.add(column)

    keep_columns = [c for c in columns if c in present_columns]
    numeric_columns = [c for c in keep_columns if c in counts and c not in not_numeric]
    means = {c: totals[c] / counts[c] for c in numeric_columns}
    means = pd.Series(means, dtype="float64").replace(
        [float("inf"), -float("inf")], float("nan")
    ).dropna().to_dict()

    sample = fix_data_types(sample.reindex(columns=keep_columns), numeric_columns)
    sample_summary = inspect_data(sample)

  #clean each chunk and stream rows into one JSON array.
    rows_written = duplicates_removed = duplicates_after = 0
    missing_after = dict.fromkeys(keep_columns, 0)
    outliers, data_types = {}, {}
    first_chunk = True
    #open the output file for writing
    with open(output_file, "w", encoding="utf-8") as output:
        output.write("[")
        with read_chunks() as reader:
            for df in reader:
                df = prepare(df)
                df = remove_empty_rows_columns(df, keep_columns)
                df = fix_data_types(df, numeric_columns)
                df = handle_missing_values(df, means, skip)
                df, detected = handle_outliers(df, sample, cap_outliers, skip)

                before = len(df)
                #remove duplicates
                if deduplicate:
                    df = remove_duplicates(df)
                duplicates_removed += before - len(df)

                quality = validate_data(df)
                duplicates_after += quality["duplicate_rows"]
                data_types = quality["data_types"]
                for column, count in quality["missing_values"].items():
                    missing_after[column] += int(count)
                for column, count in detected.items():
                    outliers[column] = outliers.get(column, 0) + count

                #add number of rows written
                rows_written += len(df)

                #convert each row to json and write to file
                text = df.to_json(
                    orient="records", force_ascii=False, double_precision=15
                )[1:-1]
            #add comma between chunks
                if text:
                    output.write(("" if first_chunk else ",") + text)
                    first_chunk = False
        #close the JSON array
        output.write("]")

  #generate final cleaning report
    return generate_report({
        "status": "success",
        "output_file": output_file,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "empty_rows_removed": empty_rows_removed,
        "empty_columns_removed": [c for c in columns if c not in present_columns],
        "duplicates_removed_within_chunks": duplicates_removed,
        "duplicates_remaining_within_chunks": duplicates_after,
        "numeric_columns": numeric_columns,
        "missing_values_after": missing_after,
        "data_types": data_types,
        "outliers_detected_before_deduplication": outliers,
        "outliers_capped": cap_outliers,
        "sample_rows": len(sample),
        "sample_summary": sample_summary,
        "notes": [
            "Duplicates are checked within each chunk.",
            "Outliers use sample IQR limits, after filling missing values.",
            "Columns with zero sample IQR are skipped.",
            "Invalid dates in configured date columns become null."
        ]
    })


if __name__ == "__main__":
    print(clean_csv("data.csv"))   

