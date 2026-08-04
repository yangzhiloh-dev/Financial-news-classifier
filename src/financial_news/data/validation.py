import pandas as pd

def remove_duplicates_and_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows and remove rows with conflicting sentiment labels
    """

    conflict_mask = df.groupby("text_hash")["label"].transform("nunique") > 1
    df = df[~conflict_mask].drop_duplicates(subset=["text_hash"]).copy()
    return df

def validate_split(no_duplicates_df: pd.DataFrame,train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Validate that the train, validation, and test splits are disjoint and cover the entire dataset.
    """
    train_hashes = set(train_df["text_hash"])
    validation_hashes = set(val_df["text_hash"])
    test_hashes = set(test_df["text_hash"])
    
    assert train_hashes.isdisjoint(validation_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert validation_hashes.isdisjoint(test_hashes)
    
    assert (
        len(train_df)
        + len(val_df)
        + len(test_df)
        == len(no_duplicates_df)
    )

