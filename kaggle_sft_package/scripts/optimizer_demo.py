import torch

from optimizers.gradient_decent import GradientDescent


def main() -> None:
    weight = torch.tensor(
        2.0,
        requires_grad=True,
    )

    target = torch.tensor(10.0)
    x = torch.tensor(2.0)

    optimizer = GradientDescent(
        parameters=[weight],
        learning_rate=0.1,
    )

    for step in range(10):
        prediction = weight * x

        loss = (prediction - target) ** 2

        optimizer.zero_grad()

        loss.backward()

        print(
            f"Step {step:2d} | "
            f"Weight={weight.item():.4f} | "
            f"Loss={loss.item():.4f} | "
            f"Grad={weight.grad.item():.4f}"
        )

        optimizer.step()


if __name__ == "__main__":
    main()
