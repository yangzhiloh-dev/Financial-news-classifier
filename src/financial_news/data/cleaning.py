import pandas as pd
import hashlib
from financial_news.config import LABEL_TO_ID

def normalise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the text and sentiment data by converting to Lowercase and removing leading/trailing whitespaces.
    """
    missing_counts = df.isna().sum()
    if missing_counts.any():
        raise ValueError(f"Found missing values in the following columns: {missing_counts[missing_counts > 0].index.tolist()}")

    df["raw_text"] = df["text"].copy()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.lower().str.strip()
    df["raw_sentiment"] = df["sentiment"].copy()
    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    return df
    

def map_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the sentiment Labels to numerical values
    """
    if not df["sentiment"].isin(LABEL_TO_ID.keys()).all():
        raise ValueError(f"Found unexpected sentiment labels: {df['sentiment'].unique()}")
    df["label"] = df["sentiment"].map(LABEL_TO_ID)

    return df
    

def make_text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def add_text_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a unique hash ID for each row based on the text content.
    """
    df["text_hash"] = df["text"].apply(make_text_hash)

    return df

def prepare_data_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_data(df)
    df = map_sentiment_labels(df)
    df = add_text_hash(df)
    df["source"] = "FinancialPhraseBank"
    df["news_id"] = "fpb_" + df["text_hash"].str[:12]
    
    return df