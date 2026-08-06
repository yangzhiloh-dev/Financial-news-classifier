"""An embedding table gives each token ID a learnable vector.

The mask removes padding.

Averaging turns all real token vectors into one document vector.

The classifier turns the document vector into three sentiment scores.

The loss measures how wrong those scores are.

Backpropagation calculates how each used vector and classifier weight
should change.

The optimizer applies those changes."""

import torch
from torch import nn

from financial_news.text.vocabulary import PAD_ID
from financial_news.config import NUM_CLASSES


def main() -> None:
    torch.manual_seed(42)

    input_ids = torch.tensor(
        [
            [2, 3, 4, 0, 0],
            [5, 6, 7, 8, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [2, 0],
        dtype=torch.long,
    )

    embedding = nn.Embedding(num_embeddings=10, embedding_dim= 4, padding_idx=PAD_ID) # num_embeddings=len(vocabulary)
    token_embeddings = embedding(input_ids)
    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    masked_embeddings = token_embeddings * expanded_mask # Mask padding embeddings

    padding_positions = attention_mask == 0
    # Verify that all padding positions become zero vectors
    assert torch.all(
        masked_embeddings[padding_positions] == 0
    )

    # Reduce into one document vector 
    summed_embeddings = masked_embeddings.sum(dim=1)

    # Count real tokens
    token_counts = expanded_mask.sum(dim=1).clamp(min=1)

    # Calculate average
    document_embeddings = summed_embeddings / token_counts

    classifier = nn.Linear(
        in_features=4, # Num of Embedding dims
        out_features=NUM_CLASSES,
    )

    logits = classifier(document_embeddings)
    print("Logits:", logits.shape)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    print("Loss:", loss)
    print("Loss shape:", loss.shape)

    optimizer = torch.optim.SGD(
        list(embedding.parameters())
        + list(classifier.parameters()),
        lr=0.1,
    )

    # Trainable parameters
    embedding.weight
    classifier.weight
    classifier.bias

    # Save a copy 
    embedding_before_update = (
        embedding.weight.detach().clone()
    )

    # Run backpropagration
    optimizer.zero_grad()

    loss.backward()

    print(
        "Embedding gradient:",
        embedding.weight.grad.shape,
    )

    print(
        "Classifier weight gradient:",
        classifier.weight.grad.shape,
    )

    print(
        "Classifier bias gradient:",
        classifier.bias.grad.shape,
    )

    assert torch.all(
        embedding.weight.grad[PAD_ID] == 0 # Check the padding gradient
    )

    assert (
        embedding.weight.grad[2].abs().sum() > 0 # Check a used token
    )

    # Update the parameters
    optimizer.step()

    embedding_after_update = (
        embedding.weight.detach().clone()
    )

    assert not torch.equal(
        embedding_before_update[2],
        embedding_after_update[2], # Check that real token changed
    )

    assert torch.equal(
        embedding_before_update[PAD_ID],
        embedding_after_update[PAD_ID], # Check that padding stayed unchanged
    )

    # Convert logits to probabilities
    probabilities = torch.softmax(
        logits.detach(),
        dim=1,
    )

    predictions = logits.detach().argmax(
        dim=1
    )

    print("Probabilities:")
    print(probabilities)

    print("Predictions:")
    print(predictions)

    print(
        "Probability sums:",
        probabilities.sum(dim=1),
    )

    # Each probability row should sum to approx 1
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(input_ids.size(0)),
    )

if __name__ == "__main__":
    main()