import typing as t
import numpy as np

from datasets import Dataset
import tqdm

from dual.typing import Tensor, LossFunction
from dual.layer import Layer
from dual.gradient import Gradient


class NeuralNetwork:
    def __init__(self, input_size: int, loss_function: LossFunction) -> None:
        self.layers: list[Layer] = []
        self.input_size = input_size
        self.loss_function = loss_function

    def add_layer(
        self, n_neurons: int, activation_function: t.Callable, initialize: str
    ) -> Layer:
        layer = Layer(
            model=self,
            n_neurons=n_neurons,
            activation_function=activation_function,
            initialize=initialize,
        )
        self.layers.append(layer)
        return layer

    def __call__(self, x: Tensor) -> Tensor:
        result = x
        for layer in self.layers:
            result = layer.compute_activation(x=result)
        return result

    def compute_loss(self, x: Tensor, y: Tensor) -> float:
        return self.loss_function(self(x), y)

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
        validation_data: Dataset | None,
        validation_frequency: int = 500,
    ) -> None:
        rolling_loss = np.full(shape=10, fill_value=np.nan)
        validation_accuracy = None

        for i in range(epochs):
            data_it = data.shuffle().iter(batch_size=1)
            with tqdm.tqdm(
                enumerate(data_it), total=data.num_rows, desc=f"EPOCH {i+1}"
            ) as it:
                for j, datapoint in it:
                    if j % validation_frequency == 0 and validation_data is not None:
                        validation_accuracy = self.accuracy(data=validation_data)

                    x = datapoint["input"][0]
                    y = datapoint["label"][0]
                    loss = self.compute_loss(x=x, y=y)
                    rolling_loss = np.roll(rolling_loss, 1)
                    rolling_loss[0] = loss
                    it.set_postfix(
                        loss=np.mean(rolling_loss),
                        validation_accuracy=validation_accuracy,
                    )
                    gradients = self.compute_gradients(x=x, y=y)
                    self.update_parameters(
                        gradients=gradients, learning_rate=learning_rate
                    )
