from torch.utils.data import DataLoader

from financial_news.config import PROCESSED_DIR
from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.text.vocabulary import Vocabulary
from financial_news.training.dataset import FinancialNewsDataset


def main() -> None:
    train_df = load_phrasebank(
        PROCESSED_DIR / "sentiment_data_train.csv"
    )

    vocabulary = Vocabulary.build(
        texts=train_df["text"],
        min_freq=2,
    )

    train_dataset = FinancialNewsDataset(
        texts=train_df["text"],
        labels=train_df["sentiment"],
        vocabulary=vocabulary,
        max_length=64,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    print("Training examples:", len(train_dataset))
    print("Vocabulary size:", len(vocabulary))
    print("Number of batches:", len(train_loader))

    print("Input shape:", batch["input_ids"].shape)
    print("Mask shape:", batch["attention_mask"].shape)
    print("Label shape:", batch["label"].shape)


if __name__ == "__main__":
    main()

