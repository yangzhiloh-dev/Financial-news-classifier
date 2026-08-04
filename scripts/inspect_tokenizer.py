from financial_news.config import PROCESSED_DIR
from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.text.tokenizer import tokenize


def inspect_random_examples() -> None:
    train_df = load_phrasebank(
        PROCESSED_DIR / "sentiment_data_train.csv"
    )

    examples = train_df.sample(
        n=10,
        random_state=42,
    )

    for _, row in examples.iterrows():
        tokens = tokenize(row["text"])

        print("\nText:")
        print(row["text"])

        print("Sentiment:")
        print(row["sentiment"], row["label"])

        print("Tokens:")
        print(tokens)

        print("Number of tokens:", len(tokens))


if __name__ == "__main__":
    inspect_random_examples()