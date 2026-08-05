import torch


def scalar_gradient_exercise() -> None:
    x = torch.tensor(
        2.0,
        requires_grad=True,
    )

    y = x**2 + 3 * x + 1

    print("x:", x)
    print("y:", y)
    print("Gradient before backward:", x.grad)

    y.backward()

    print("Gradient after backward:", x.grad)

    expected_gradient = torch.tensor(7.0)

    assert torch.isclose(
        x.grad,
        expected_gradient,
    )

def gradient_accumulation_exercise() -> None:
    parameter = torch.tensor(
        2.0,
        requires_grad=True,
    )

    first_loss = parameter**2
    first_loss.backward()

    print("After first backward:", parameter.grad)
    # 4

    second_loss = parameter**2
    second_loss.backward()

    print("After second backward:", parameter.grad)
    # 8

    parameter.grad.zero_()

    print("After zeroing:", parameter.grad)
    # 0

def manual_optimization_exercise() -> None:
    weight = torch.tensor(
        0.0,
        requires_grad=True,
    )

    input_value = torch.tensor(2.0)
    target = torch.tensor(10.0)

    learning_rate = 0.1

    # Gradient descent
    for step in range(10):
        prediction = weight * input_value

        loss = (prediction - target) ** 2

        loss.backward()

        with torch.no_grad():
            weight -= learning_rate * weight.grad

        weight.grad.zero_()

        print(
            f"Step {step + 1}: "
            f"weight={weight.item():.4f}, "
            f"loss={loss.item():.4f}"
        )


def main() -> None:
    scalar_gradient_exercise()
    gradient_accumulation_exercise()
    manual_optimization_exercise()


if __name__ == "__main__":
    main()