import torch
from torch import nn

from financial_news.text.vocabulary import PAD_ID


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

    embeddings = nn.Embedding(num_embeddings=10, embedding_dim= 4, padding_idx=PAD_ID)
    token_embeddings = embeddings(input_ids)
    expanded_mask = 
    expanded_mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
    masked_embeddings = embeddings * expanded_mask # Mask padding embeddings

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