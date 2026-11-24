import json
from pathlib import Path
from time import perf_counter

from src.data_mining.cleaning import clean_dataset
from src.data_mining.profiler import profile_loaded_dataset


root = Path.home() / "MINER-benchmarks"
storage = root / "datasets"
parent_id = (root / "dataset_id.txt").read_text(encoding="utf-8").strip()
manifest = json.loads(
    (storage / parent_id / "manifest.json").read_text(encoding="utf-8")
)

# All columns together define an exact duplicate for this benchmark.
steps = [{"operation": "drop_duplicates", "columns": manifest["columns"]}]

print("Cleaning 1,000,000 rows...", flush=True)
start = perf_counter()
result = clean_dataset(storage, parent_id, steps)
elapsed = perf_counter() - start

# Independently profile the saved child to check what actually reached disk.
profile = profile_loaded_dataset(
    storage, result["dataset_id"], engine="pandas", include_correlations=False
)

assert result["parent_dataset_id"] == parent_id
assert result["rows_before"] == 1_000_000
assert result["changes"][0]["rows_removed"] == 2_128
assert result["rows_after"] == profile["dataset"]["rows"] == 997_872
assert profile["dataset"]["duplicate_rows"] == 0

print(f"Cleaning time: {elapsed:.1f} seconds")
print(f"Cleaned dataset ID: {result['dataset_id']}")
print("PASS: 2,128 rows removed; 997,872 rows saved; 0 duplicates remain.")