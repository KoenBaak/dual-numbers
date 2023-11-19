import typing as t

import numpy as np
import numpy.typing as npt

from dual.structure_decorators import (
    commutative_multiplication,
    commutative_addition,
    subtract_using_negative,
)
from dual.number import DualNumber


_POINTWISE_DERIVATIVES = {}
_HANDLED_UFUNCS = {}


def implements(numpy_func: t.Callable) -> t.Callable:
    def decorator(f):
        _HANDLED_UFUNCS[numpy_func] = f
        return f

    return decorator


def pointwise_derivative(ufunc: np.ufunc) -> t.Callable:
    def decorator(f):
        _POINTWISE_DERIVATIVES[ufunc] = f
        return f

    return decorator


class _DualTensorRepr:
    def __init__(self, tensor: "DualTensor") -> None:
        self._tensor = tensor
        self.shape = tensor.real.shape
        self.size = tensor.real.size
        self.ndim = tensor.real.ndim

    def __getitem__(self, item: t.Any) -> DualNumber:
        return self._tensor[item]

    def __repr__(self) -> str:
        return np.array2string(self)


def dual_sum(a: "DualTensor", axis=0, dtype=None):
    return DualTensor(real=a.real.sum(axis=axis), dual=a.dual.sum(axis=axis))


@commutative_addition
@commutative_multiplication
@subtract_using_negative
class DualTensor:
    def __init__(
        self,
        real: npt.ArrayLike | None = None,
        dual: npt.ArrayLike | None = None,
        dtype: npt.DTypeLike = np.float_,
    ) -> None:
        assert real is not None or dual is not None
        assert np.dtype(dtype=dtype).kind == "f"

        self.real = (
            np.asarray(real, dtype=dtype)
            if real is not None
            else np.zeros_like(dual, dtype=dtype)
        )
        self.dual = (
            np.asarray(dual, dtype=dtype)
            if dual is not None
            else np.zeros_like(real, dtype=dtype)
        )

        assert self.real.shape == self.dual.shape

    @classmethod
    def with_dual_ones(
        cls, real: npt.NDArray, dtype: npt.DTypeLike = np.float_
    ) -> "DualTensor":
        return DualTensor(real=real, dual=np.ones_like(real, dtype=dtype), dtype=dtype)

    def __getitem__(self, item: t.Any) -> DualNumber:
        match item:
            case tuple():
                return DualNumber(real=self.real[item], dual=self.dual[item])
            case _:
                return NotImplemented

    def __add__(self, other: t.Any) -> "DualTensor":
        match other:
            case DualTensor() | DualNumber():
                return DualTensor(
                    real=self.real + other.real, dual=self.dual + other.dual
                )
            case float() | int() | np.ndarray():
                return DualTensor(real=self.real + other, dual=self.dual)
            case _:
                return NotImplemented

    def __mul__(self, other: t.Any) -> "DualTensor":
        match other:
            case DualTensor() | DualNumber():
                return DualTensor(
                    real=self.real * other.real,
                    dual=self.real * other.dual + other.real * self.dual,
                )
            case float() | int() | np.ndarray():
                return DualTensor(real=self.real * other, dual=self.dual * other)
            case _:
                return NotImplemented

    def __neg__(self) -> "DualTensor":
        return DualTensor(real=-self.real, other=-self.other)

    def __pow__(self, power, modulo=None) -> "DualTensor":
        match power:
            case int():
                return DualTensor(
                    real=self.real**power,
                    dual=self.dual * power * self.real ** (power - 1),
                )

    def __truediv__(self, other):
        match other:
            case DualTensor():
                inv = DualTensor(
                    real=1 / other.real, dual=-other.dual / (other.real**2)
                )
            case _:
                inv = 1 / other
        return self * inv

    @staticmethod
    def pointwise_function(func: np.ufunc, x: "DualTensor", extra_args: tuple) -> t.Any:
        derivative = _POINTWISE_DERIVATIVES.get(func)
        if derivative is None:
            return NotImplemented
        return DualTensor(
            real=func(x.real, *extra_args),
            dual=derivative(x.real, *extra_args) * x.dual,
        )

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *args, **kwargs) -> t.Any:
        if method == "reduce" and ufunc == np.add:
            return dual_sum(*args, **kwargs)

        if method != "__call__":
            return NotImplemented

        handler = _HANDLED_UFUNCS.get(ufunc)
        if handler is not None:
            return handler(*args, **kwargs)

        if ufunc.signature is None:
            return self.pointwise_function(func=ufunc, x=args[0], extra_args=args[1:])

        return NotImplemented

    def __repr__(self) -> str:
        return f"DualTensor(\n{_DualTensorRepr(self)}\n)"


from dual.implementations import *
