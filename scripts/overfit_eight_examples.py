# The goal is to make the model deliberately memorise eight examples.
# If a model with 500,000 parameters cannot memorise eight examples,
# there may be a problem with:
# - inputs or labels
# - vocabulary or attention masks
# - forward pass
# - loss calculation
# - backpropagation
# - optimizer
# - training-loop order
# After repeated training, traiing accuracy should be equal 100%, training loss should be close to 0

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from financial_news.config import PROCESSED_DIR, NUM_CLASSES
from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.text.vocabulary import Vocabulary, PAD_ID
from financial_news.training.dataset import FinancialNewsDataset
from financial_news.models.embedding_mlp import EmbeddingAverageMLP


def select_eight_examples(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    requested_examples = {
        "negative": 3,
        "neutral": 3,
        "positive": 2,
    }

    selected_parts = []

    for seed, (sentiment, count) in enumerate(
        requested_examples.items(),
        start=42,
    ):
        candidates = train_df[
            train_df["sentiment"] == sentiment
        ]

        if len(candidates) < count:
            raise ValueError(
                f"Not enough {sentiment} examples."
            )

        selected_parts.append(
            candidates.sample(
                n=count,
                random_state=seed,
            )
        )

    tiny_df = (
        pd.concat(selected_parts)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    if len(tiny_df) != 8:
        raise RuntimeError(
            "Expected exactly eight examples."
        )

    return tiny_df


def calculate_accuracy(
        logits: torch.Tensor, 
        labels: torch.Tensor
) -> float:
    if logits.size(0) != labels.size(0):
        raise ValueError(
            "Logits and labels must contain the same number of examples."
        )

    predictions = logits.argmax(dim=1) # search horizontally across columns for each individual row to find the index of the max
    return (predictions == labels).float().mean().item()


def train_until_memorized(
        model: EmbeddingAverageMLP, 
        train_loader: DataLoader , 
        max_epochs: int = 500, 
        loss_threshold: float = 0.01,
) -> tuple[float, float]:

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    final_loss = float("inf")
    final_accuracy = 0.0
    memorized = False

    for epoch in range(1, max_epochs + 1):
        model.train()

        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            final_accuracy = calculate_accuracy(logits=logits, labels=labels)

            if epoch == 1 or epoch % 25 == 0:
                print(
                f"Epoch {epoch:3d} | "
                f"Loss: {final_loss:.6f} | "
                f"Accuracy: "
                f"{final_accuracy:.2%}"
            )
            if (final_accuracy == 1 and final_loss < loss_threshold):
                memorized=True
                print(
                    f"Memorized eight examples at "
                    f"epoch {epoch}."
                )
            if memorized:
                break
        if memorized:
            break
    if not memorized:
        raise RuntimeError(f"Model failed to memorise 8 examples in {max_epochs} epochs")

    return ( final_accuracy, final_loss )


def evaluate_memorization(
        model: EmbeddingAverageMLP, 
        evaluation_loader: DataLoader
) -> None:
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        batch = next(iter(evaluation_loader))

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        accuracy = calculate_accuracy(
            logits,
            labels,
        )

        print("\nFinal evaluation:")
        print("Targets:    ", labels.tolist())
        print("Predictions:", predictions.tolist())
        print("Loss:", loss.item())
        print("Accuracy:", f"{accuracy:.2%}")

        assert accuracy == 1.0
        assert loss.item() < 0.01
        assert torch.equal(predictions, labels)


def main() -> None:
    torch.manual_seed(42)

    train_df = load_phrasebank(PROCESSED_DIR / "sentiment_data_train.csv")
    texts = train_df["text"]

    vocabulary = Vocabulary.build(texts)

    tiny_df = select_eight_examples(train_df)
    print(tiny_df[["text", "sentiment"]])
    print(tiny_df['sentiment'].value_counts())

    tiny_dataset = FinancialNewsDataset(texts=tiny_df["text"],
                                   labels=tiny_df["sentiment"],
                                   vocabulary=vocabulary
                                )

    train_dataloader = DataLoader(tiny_dataset, 
                                  batch_size=8, 
                                  shuffle=True
                                  )
    eval_dataloader = DataLoader(tiny_dataset,
                                 batch_size=8,
                                 shuffle=False
                                 )

    model = EmbeddingAverageMLP(vocabulary_size=len(vocabulary),
                                embedding_dimension=64,
                                hidden_dimension=64,
                                num_of_classes=NUM_CLASSES,
                                padding_id=PAD_ID,
                                dropout_probability=0.0
                                )

    train_until_memorized(model=model,
                          train_loader=train_dataloader
                          )

    evaluate_memorization(model=model, 
                          evaluation_loader=eval_dataloader
                          )

    print("Memorization success")

if __name__ == "__main__":
    main()