import numpy as np
from dual.tensor import pointwise_derivative, implements, DualTensor


@pointwise_derivative(np.sin)
def der_sin(x):
    return np.cos(x)


@pointwise_derivative(np.log)
def der_log(x):
    return 1 / x


@pointwise_derivative(np.cos)
def der_cos(x):
    return -np.sin(x)


@pointwise_derivative(np.exp)
def der_exp(x):
    return np.exp(x)


@pointwise_derivative(np.sqrt)
def der_sqrt(x):
    return 1 / (2 * np.sqrt(x))


@pointwise_derivative(np.maximum)
def der_maximum(a, b):
    return np.where(a > b, 1, 0)


@implements(np.matmul)
def dual_matmul(A, x):
    if isinstance(A, DualTensor) and isinstance(x, DualTensor):
        return NotImplemented
    elif isinstance(A, DualTensor):
        return DualTensor(A.real @ x, A.dual @ x)
    else:
        return DualTensor(A @ x.real, A @ x.dual)


@implements(np.add)
def dual_add(a, b):
    if isinstance(a, DualTensor):
        return a.__add__(b)

    return b.__add__(a)


@implements(np.multiply)
def dual_add(a, b):
    if isinstance(a, DualTensor):
        return a.__mul__(b)

    return b.__mul__(a)
