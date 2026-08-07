import torch
from torch import nn

def masked_mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor,) -> torch.Tensor:
    """
    Reduces token_embeddings into one document vector, reducing it by one dimension
    """

    if token_embeddings.ndim != 3:
        raise ValueError(
            "token_embeddings must have shape "
            "[batch, sequence, embedding]."
        )

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape "
            "[batch, sequence]."
        )

    if token_embeddings.shape[:2] != attention_mask.shape:
        raise ValueError(
            "The batch and sequence dimensions must match."
        )

    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    masked_embeddings = expanded_mask * token_embeddings

    padding_positions = attention_mask == 0
    # Verify that all padding positions become zero vectors
    assert torch.all(
        masked_embeddings[padding_positions] == 0
    )

    # Reducing into each document into one vector
    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1)
    document_embeddings = summed_embeddings / token_counts

    return document_embeddings

class EmbeddingAverageMLP(nn.Module):
    def __init__(self, 
                 vocabulary_size: int,
                 embedding_dimension:int,
                 hidden_dimension: int,
                 num_of_classes: int,
                 padding_id: int,
                 dropout_probability: float,
    ) -> None:
        super().__init__()

        if vocabulary_size <= 2:
            raise ValueError("vocabulary size must be greater than 2 to include padding and unknown tokens")

        if embedding_dimension < 1:
            raise ValueError("Embedding dimension must be a positive number")

        if hidden_dimension < 1:
            raise ValueError("hidden dimension must be a positive number")

        if num_of_classes < 2:
            raise ValueError("Number of classes must be at least 2")

        if not 0 <= dropout_probability < 1:
            raise ValueError("Dropout probability must be between 0 and 1")

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id
        )

        self.hidden_layer = nn.Linear(
            in_features=embedding_dimension,
            out_features=hidden_dimension,             
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_probability)
        self.output_layer = nn.Linear(
            in_features= hidden_dimension,
            out_features= num_of_classes
        )

        self.classifier = nn.Sequential(
            self.hidden_layer,
            self.activation,
            self.dropout,
            self.output_layer
        )


    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert token_ids into raw sentiment logits
        """

        if not input_ids.ndim == 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        if not attention_mask.shape == input_ids.shape:
            raise ValueError("input_ids and attention_mask must have same shape")

        if not input_ids.dtype == torch.long: 
            raise TypeError("input_ids must use torch long")   

        token_embeddings = self.embedding(input_ids)

        document_embeddings = masked_mean_pool(token_embeddings=token_embeddings, attention_mask=attention_mask)

        logits = self.classifier(document_embeddings)

        return logits