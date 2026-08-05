import torch


def main() -> None:
    # Create tensors
    scalar = torch.tensor(4.0)

    vector = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    print("Scalar shape:", scalar.shape)
    print("Vector shape:", vector.shape)
    print("Matrix shape:", matrix.shape)

    print("Matrix shape:", matrix.shape) # Size along each dim
    print("Matrix dimensions:", matrix.ndim) # Num of dims
    print("Matrix dtype:", matrix.dtype) # type of values stored

    # Create a model batch
    input_ids = torch.tensor(
        [
            [12, 45, 83, 0, 0],
            [37, 91, 24, 18, 0],
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

    print("Input shape:", input_ids.shape)
    print("Mask shape:", attention_mask.shape)
    print("Label shape:", labels.shape)

    # Indexing
    print("First document:", input_ids[0])
    print("First token of first document:", input_ids[0, 0])
    print("First three tokens:", input_ids[0, :3])
    print("First label:", labels[0])

    values = torch.arange(24)
    reshaped = values.reshape(2, 3, 4)
    # Reshaping dims
    print(values.shape)
    print(reshaped.shape)   

    mask_for_embeddings = attention_mask.unsqueeze(-1)
    # Adding dims
    print(attention_mask.shape)
    print(mask_for_embeddings.shape)

    # Create fake embeddings
    embeddings = torch.randn(
        2,  # batch size
        5,  # sequence length
        4,  # embedding dimension
    )

    mask_for_embeddings = (
        attention_mask
        .unsqueeze(-1)
        .to(embeddings.dtype)
    )

    masked_embeddings = (
        embeddings * mask_for_embeddings
    )

    print("Embeddings:", embeddings.shape)
    print("Expanded mask:", mask_for_embeddings.shape)
    print("Result:", masked_embeddings.shape)

    # Reductions
    summed_embeddings = masked_embeddings.sum(dim=1)
    # Sum across the sequence dimension
    print(summed_embeddings.shape)

    token_counts = (
        attention_mask
        .sum(dim=1, keepdim=True)
        .clamp(min=1)
    )

    print(token_counts)
    print(token_counts.shape)

    average_embeddings = (
        summed_embeddings / token_counts
    )

    print(average_embeddings.shape)

    # Matrix multiplication
    features = torch.randn(2, 4)
    weights = torch.randn(4, 3)
    bias = torch.randn(3)

    logits = features @ weights + bias

    # Expected: [2, 3]
    # 2 Examples, 3 sentiment scores per example
    print(logits.shape)

if __name__ == "__main__":
    main()