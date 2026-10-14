import pandas as pd

# Profiler public interface i.e Public API

def profile_dataset(df: pd.DataFrame) -> dict:
    """Validate input type"""
    if not isinstance(df , pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")

    return {
        "dataset" : {
            "rows" : int(df.shape[0]),
            "columns" : int(df.shape[1])
        }
    }