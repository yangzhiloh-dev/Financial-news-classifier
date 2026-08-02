import pandas as pd
from sklearn.model_selection import train_test_split

def split_sentiment_data(df: pd.DataFrame, test_size = 0.15, val_size = 0.15, random_state=42):
    """
    split the data into train, validation, and test sets based on the specified sizes.
    """
    if not (0 < test_size < 1) or not (0 < val_size < 1):
        raise ValueError("test_size and val_size must be between 0 and 1.")

    if test_size + val_size >= 1:
        raise ValueError("The sum of test_size and val_size must be less than 1.")

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state, stratify=df["label"])

    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size+val_size), random_state=random_state, stratify=temp_df["label"])

    return (
    train_df.reset_index(drop=True),
    val_df.reset_index(drop=True),
    test_df.reset_index(drop=True),
    )

def split_x_y(df: pd.DataFrame):
    """
    Split the DataFrame into features (X) and Labels (y) for model training and evaluation.
    """
    X = df["text"].copy()
    y = df["label"].copy()
    return X, y