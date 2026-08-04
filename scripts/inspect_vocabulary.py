from financial_news.config import PROCESSED_DIR
from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.text.vocabulary import (
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
)


def main() -> None:
    train_df = load_phrasebank(
        PROCESSED_DIR / "sentiment_data_train.csv"
    )

    vocabulary = Vocabulary.build(
        texts=train_df["text"],
        min_freq=2,
        max_size=None,
    )

    print("Vocabulary size:", len(vocabulary))
    print("PAD ID:", vocabulary.token_to_id[PAD_TOKEN])
    print("UNK ID:", vocabulary.token_to_id[UNK_TOKEN])

    example_text = train_df.iloc[0]["text"]
    tokens_as_ids = vocabulary.encode_text(example_text)
    decoded_tokens = vocabulary.decode_ids(tokens_as_ids)

    print("\nOriginal text:")
    print(example_text)

    print("\nToken IDs:")
    print(tokens_as_ids)

    print("\nDecoded tokens:")
    print(decoded_tokens)

    unknown_token = "this-token-was-never-in-training"

    print(
        "\nUnknown token ID:",
        vocabulary.lookup_token(unknown_token),
    )


if __name__ == "__main__":
    main()