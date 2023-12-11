from dataclasses import dataclass
from datasets import Dataset


@dataclass
class Gradient:
    weights: Tensor
    bias: Tensor


class Layer:
    ...

    def push_forward(
        self,
        x: Tensor,
        y: Tensor,
        weights: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> Tensor:
        x = self.compute_activation(x=x, weights=weights, bias=bias)
        if self.next_layer is not None:
            return self.next_layer.push_forward(x=x, y=y)

        return self.model.loss_function(x, y)

    def compute_gradient(self, x: Tensor, y: Tensor) -> Gradient:
        weights_gradient = np.zeros_like(self.weights)
        with np.nditer(self.weights, flags=["multi_index"]) as it:
            for _ in it:
                dual = np.zeros_like(self.weights)
                dual[it.multi_index] = 1
                dual_parameter = DualTensor(real=self.weights, dual=dual)

                weights_gradient[it.multi_index] = self.push_forward(
                    x=x, y=y, weights=dual_parameter
                ).dual

        bias_gradient = np.zeros_like(self.bias)
        for i in range(self.n_neurons):
            dual = np.zeros_like(self.bias)
            dual[i] = 1
            dual_parameter = DualTensor(real=self.bias, dual=dual)
            bias_gradient[i] = self.push_forward(x=x, y=y, bias=dual_parameter).dual

        return Gradient(weights=weights_gradient, bias=bias_gradient)

    def update_parameters(self, gradient: Gradient, learning_rate: float) -> None:
        self.weights = self.weights - learning_rate * gradient.weights
        self.bias = self.bias - learning_rate * gradient.bias


class NeuralNetwork:
    ...

    def compute_gradients(self, x: Tensor, y: Tensor) -> list[Gradient]:
        gradients = []
        for layer in self.layers:
            gradients.append(layer.compute_gradient(x=x, y=y))
            x = layer.compute_activation(x=x)
        return gradients

    def update_parameters(
        self, gradients: list[Gradient], learning_rate: float
    ) -> None:
        for i, gradient in enumerate(gradients):
            self.layers[i].update_parameters(gradient, learning_rate=learning_rate)

    def accuracy(self, data: Dataset) -> float:
        it = data.iter(batch_size=1)
        correct = 0
        for d in it:
            x = d["input"][0]
            y = d["label"][0]
            y_pred = np.argmax(self(x))
            correct += y_pred == np.argmax(y)
        return correct / data.num_rows

    def train(
        self,
        data: Dataset,
        epochs: int,
        learning_rate: float,
    ) -> None:
        for i in range(epochs):
            for datapoint in data.shuffle().iter(batch_size=1):
                x = datapoint["input"][0]
                y = datapoint["label"][0]
                gradients = self.compute_gradients(x=x, y=y)
                self.update_parameters(gradients=gradients, learning_rate=learning_rate)
