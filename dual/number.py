import typing as t

from dual.structure_decorators import (
    commutative_addition,
    commutative_multiplication,
    subtract_using_negative,
)


@subtract_using_negative
@commutative_multiplication
@commutative_addition
class DualNumber:
    def __init__(self, real: float = 0.0, dual: float = 0.0) -> None:
        self.real = real
        self.dual = dual

    def __repr__(self) -> str:
        return f"{self.real} + {self.dual}ε"

    def __add__(self, other: t.Any) -> "DualNumber":
        match other:
            case DualNumber():
                return DualNumber(
                    real=self.real + other.real, dual=self.dual + other.dual
                )
            case float() | int():
                return DualNumber(real=self.real + other, dual=self.dual)
            case _:
                return NotImplemented

    def __mul__(self, other: t.Any) -> "DualNumber":
        match other:
            case DualNumber():
                return DualNumber(
                    real=self.real * other.real,
                    dual=self.real * other.dual + other.real * self.dual,
                )
            case float() | int():
                return DualNumber(real=self.real * other, dual=self.dual * other)
            case _:
                return NotImplemented

    def __neg__(self) -> "DualNumber":
        return self.__mul__(-1)
