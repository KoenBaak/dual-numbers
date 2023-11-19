from dataclasses import dataclass
from dual.typing import Tensor


@dataclass
class Gradient:
    weights: Tensor
    bias: Tensor
