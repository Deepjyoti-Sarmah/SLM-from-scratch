import torch


class GradientDescent:
    def __init__(
        self,
        *,
        parameters,
        learning_rate: float,
    ) -> None:
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def step(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is None:
                    continue

                parameter -= self.learning_rate * parameter.grad

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None
