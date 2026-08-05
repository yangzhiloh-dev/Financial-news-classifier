import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Create synthetic data
def create_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(42)

    number_of_examples = 200

    features = torch.randn(
        number_of_examples,
        1,
    )

    noise = 0.1 * torch.randn(
        number_of_examples,
        1,
    )

    targets = 3 * features + 2 + noise

    shuffled_indices = torch.randperm(
        number_of_examples
    )

    training_indices = shuffled_indices[:160]
    validation_indices = shuffled_indices[160:]

    return (
        features[training_indices],
        targets[training_indices],
        features[validation_indices],
        targets[validation_indices],
    )

# Define a nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=1,
            out_features=1,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(features)

def create_train_loader(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> DataLoader:
    dataset = TensorDataset(
        features,
        targets,
    )

    generator = torch.Generator().manual_seed(42)

    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=generator,
    )

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    number_of_epochs: int = 100,
) -> None:
    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    for epoch in range(1, number_of_epochs + 1):
        model.train()

        total_training_loss = 0.0

        for features, targets in train_loader:
            optimizer.zero_grad()

            predictions = model(features)

            loss = criterion(
                predictions,
                targets,
            )

            loss.backward()

            optimizer.step()

            total_training_loss += (
                loss.item() * features.size(0)
            )

        average_training_loss = (
            total_training_loss
            / len(train_loader.dataset)
        )

        if epoch == 1 or epoch % 20 == 0:
            model.eval()

            with torch.no_grad():
                validation_predictions = model(
                    validation_features
                )

                validation_loss = criterion(
                    validation_predictions,
                    validation_targets,
                )

            print(
                f"Epoch {epoch:3d} | "
                f"Train loss: "
                f"{average_training_loss:.6f} | "
                f"Validation loss: "
                f"{validation_loss.item():.6f}"
            )

def main() -> None:
    (
        train_features,
        train_targets,
        validation_features,
        validation_targets,
    ) = create_data()

    train_loader = create_train_loader(
        train_features,
        train_targets,
    )

    model = LinearRegressionModel()

    train_model(
        model=model,
        train_loader=train_loader,
        validation_features=validation_features,
        validation_targets=validation_targets,
    )

    learned_weight = model.linear.weight.item()
    learned_bias = model.linear.bias.item()

    print("\nTrue weight: 3.0")
    print("Learned weight:", learned_weight)

    print("True bias: 2.0")
    print("Learned bias:", learned_bias)

    assert abs(learned_weight - 3.0) < 0.2
    assert abs(learned_bias - 2.0) < 0.2


if __name__ == "__main__":
    main()
