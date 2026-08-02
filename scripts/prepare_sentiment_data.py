from financial_news.config import PROCESSED_DIR, RAW_DIR
from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.data.validation import remove_duplicates_and_conflicts, validate_split
from financial_news.data.cleaning import prepare_data_for_model
from financial_news.data.splitting import split_sentiment_data

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def prepare_sentiment_data() -> None:
    """
    Prepare the sentiment data by loading dataset, normalising text and saving the processed data to a CSV file.
    """

    df = load_phrasebank(RAW_DIR / "all-data.csv")

    df = prepare_data_for_model(df)
    no_duplicates_df = remove_duplicates_and_conflicts(df)

    df.to_csv(PROCESSED_DIR / "sentiment_data.csv", index=False, encoding="utf-8")
    no_duplicates_df.to_csv(PROCESSED_DIR / "sentiment_data_no_duplicates.csv", index=False, encoding="utf-8")

    train_df, val_df, test_df = split_sentiment_data(no_duplicates_df)

    validate_split(no_duplicates_df, train_df, val_df, test_df)

    train_df.to_csv(PROCESSED_DIR / "sentiment_data_train.csv", index=False, encoding="utf-8")
    val_df.to_csv(PROCESSED_DIR / "sentiment_data_val.csv", index=False, encoding="utf-8")
    test_df.to_csv(PROCESSED_DIR / "sentiment_data_test.csv", index=False, encoding="utf-8")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name} : {len(split_df)} rows")
        print(split_df["label"].value_counts())

if __name__ == "__main__":
    prepare_sentiment_data()