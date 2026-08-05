from torch.utils.data import DataLoader

from financial_news.config import PROCESSED_DIR
from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.text.vocabulary import Vocabulary
from financial_news.training.dataset import FinancialNewsDataset


def main() -> None:
    train_df = load_phrasebank(
        PROCESSED_DIR / "sentiment_data_train.csv"
    )

    val_df = load_phrasebank(
        PROCESSED_DIR / "sentiment_data_val.csv"
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

    val_dataset = FinancialNewsDataset(
            texts=val_df["text"],
            labels=val_df["sentiment"],
            vocabulary=vocabulary,
            max_length=64,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("Training dataset:")
    print("Training examples:", len(train_dataset))
    print("Vocabulary size:", len(vocabulary))
    print("Number of batches:", len(train_loader))

    print("Input shape:", train_batch["input_ids"].shape)
    print("Mask shape:", train_batch["attention_mask"].shape)
    print("Label shape:", train_batch["label"].shape)

    print("Validation dataset:")
    print("Validation examples:", len(val_dataset))
    print("Vocabulary size:", len(vocabulary))
    print("Number of batches:", len(val_loader))
    
    print("Input shape:", val_batch["input_ids"].shape)
    print("Mask shape:", val_batch["attention_mask"].shape)
    print("Label shape:", val_batch["label"].shape)


if __name__ == "__main__":
    main()

