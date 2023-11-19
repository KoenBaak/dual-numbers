import typing as t
import numpy.typing as npt
from dual.tensor import DualTensor


Tensor: t.TypeAlias = npt.NDArray[float] | DualTensor
LossFunction: t.TypeAlias = t.Callable[[Tensor, Tensor], float]
