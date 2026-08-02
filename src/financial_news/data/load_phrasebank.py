import pandas as pd

def load_phrasebank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    expected_columns = {"sentiment", "text"}

    if not expected_columns.issubset(df.columns):
        raise ValueError(f"Expected columns {expected_columns}, found {df.columns}")

    return df
