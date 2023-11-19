import typing as t
import numpy as np
import numpy.typing as npt

from dual.tensor import DualTensor
from dual.typing import Tensor
from dual.gradient import Gradient

if t.TYPE_CHECKING:
    from dual.neural_net import NeuralNetwork


class Layer:
    def __init__(
        self,
        model: "NeuralNetwork",
        n_neurons: int,
        activation_function: t.Callable,
        initialize: str = "Xavier",
    ) -> None:
        self.model = model
        self.prev_layer = model.layers[-1] if model.layers else None
        if self.prev_layer is not None:
            self.prev_layer.next_layer = self
        self.next_layer = None
        self.n_neurons = n_neurons
        self.initialization = initialize
        self.weights = self.initial_weights()
        self.bias = np.zeros(shape=self.n_neurons)
        self.activation_function = activation_function

    def initial_weights(self) -> npt.NDArray[float]:
        shape = (self.n_neurons, self.input_size)
        match self.initialization:
            case "Xavier":
                return np.random.uniform(low=-1, high=1, size=shape) * np.sqrt(
                    6 / (self.input_size + self.n_neurons)
                )
            case "He":
                return np.random.normal(
                    loc=0.0,
                    scale=np.sqrt(2 / self.input_size),
                    size=shape,
                )
            case _:
                return NotImplemented

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
