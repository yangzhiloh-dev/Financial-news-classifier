import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from financial_news.config import PROCESSED_DIR
from financial_news.text.vocabulary import PAD_ID, Vocabulary
from financial_news.config import NUM_CLASSES
from financial_news.models.embedding_mlp import EmbeddingAverageMLP
from financial_news.training.dataset import FinancialNewsDataset

def main() -> None:
    train_df = pd.read_csv( PROCESSED_DIR / "sentiment_data_train.csv" )

    vocabulary = Vocabulary.build(train_df["text"], min_freq=2, max_size=None)

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

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"]

    model = EmbeddingAverageMLP(vocabulary_size=len(vocabulary), 
                                embedding_dimension=128, 
                                hidden_dimension=64, 
                                num_of_classes=NUM_CLASSES, 
                                padding_id=PAD_ID, 
                                dropout_probability=0.2
            )

    # Disable dropout for a deterministic inspection
    model.eval()

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    print("Dataset size:", len(train_dataset))
    print("Vocabulary size:", len(vocabulary))

    print("\nBatch shapes:")
    print("Input IDs:", input_ids.shape)
    print("Attention mask:", attention_mask.shape)
    print("Labels:", labels.shape)

    print("\nModel output:")
    print("Logits:", logits.shape)

    # Count every trainable number in the model.
    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        number_of_parameters,
    )

    # Verify expected batch shapes.
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)

    assert input_ids.shape == torch.Size(
        [batch_size, sequence_length]
    )

    assert attention_mask.shape == input_ids.shape

    assert labels.shape == torch.Size([batch_size])

    assert logits.shape == torch.Size(
        [batch_size, NUM_CLASSES]
    )

    # Verify tensor data types.
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
    assert labels.dtype == torch.long
    assert logits.dtype.is_floating_point

    # Verify that the model did not produce NaN or infinity.
    assert torch.isfinite(logits).all()

     # Check that logits are compatible with classification loss.
    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    print("Loss:", loss.item())
    print("Loss shape:", loss.shape)

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)

     # Convert logits into probabilities for inspection only.
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    assert probabilities.shape == logits.shape

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(batch_size),
        atol=1e-6,
    )

    print("\nFirst example:")
    print("Label:", labels[0].item())
    print("Logits:", logits[0])
    print("Probabilities:", probabilities[0])

    print("\nAll model inspection checks passed.")


if __name__ == "__main__":
    main()



