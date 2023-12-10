import typing as t
from enum import Enum
import numpy as np
import numpy.typing as npt

from dual.tensor import DualTensor


Tensor: t.TypeAlias = npt.NDArray[float] | DualTensor
LossFunction: t.TypeAlias = t.Callable[[Tensor, Tensor], float]


class Initialize(str, Enum):
    XAVIER = "Xavier"
    HE = "He"


class Layer:
    def __init__(
        self,
        model: "NeuralNetwork",
        n_neurons: int,
        activation_function: t.Callable,
        initialize: Initialize = "Xavier",
    ) -> None:
        self.model = model
        self.prev_layer = model.layers[-1] if model.layers else None
        if self.prev_layer is not None:
            self.prev_layer.next_layer = self
        self.next_layer = None
        self.n_neurons = n_neurons
        self.initialization = Initialize(initialize)
        self.weights = self.initial_weights()
        self.bias = np.zeros(shape=self.n_neurons)
        self.activation_function = activation_function

    def initial_weights(self) -> npt.NDArray[float]:
        shape = (self.n_neurons, self.input_size)
        match self.initialization:
            case Initialize.XAVIER:
                return np.random.uniform(low=-1, high=1, size=shape) * np.sqrt(
                    6 / (self.input_size + self.n_neurons)
                )
            case Initialize.HE:
                return np.random.normal(
                    loc=0.0,
                    scale=np.sqrt(2 / self.input_size),
                    size=shape,
                )

    @property
    def input_size(self) -> int:
        return self.prev_layer.n_neurons if self.prev_layer else self.model.input_size

    def compute_activation(
        self,
        x: Tensor,
        weights: Tensor | None = None,
        bias: Tensor | None = None,
    ) -> Tensor:
        weights = weights if weights is not None else self.weights
        bias = bias if bias is not None else self.bias
        return self.activation_function(weights @ x + bias)


class NeuralNetwork:
    def __init__(self, input_size: int, loss_function: LossFunction) -> None:
        self.layers: list[Layer] = []
        self.input_size = input_size
        self.loss_function = loss_function

    def add_layer(
        self, n_neurons: int, activation_function: t.Callable, initialize: Initialize
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
