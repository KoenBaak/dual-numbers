import typing as t
import numpy as np

from dual import DualNumber, eps


def dual_sin(x: DualNumber) -> DualNumber:
    return np.sin(x.real) + np.cos(x.real) * x.dual * eps


def dual_exp(x: DualNumber) -> DualNumber:
    return np.exp(x.real) + np.exp(x.real) * x.dual * eps


def my_func(x: DualNumber) -> DualNumber:
    return dual_exp(dual_sin(x)) * 3 + dual_sin(x) * dual_sin(x)


def compute_derivative(f: t.Callable[[DualNumber], DualNumber], x: float) -> float:
    return f(x + eps).dual


compute_derivative(my_func, np.pi)
