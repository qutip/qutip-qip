from typing import Sequence, TypeAlias

import numpy as np

__all__ = [
    "Int", "Real", "Number", "ScalarList", "IntList", "RealList", "ScalarList", "ArrayLike",
]

# TODO When minimum version is updated to 3.12, use type (PEP 695) in place of TypeAlias

Int: TypeAlias = int | np.integer
Real: TypeAlias = int | float | np.integer | np.floating
Number: TypeAlias = int | float | complex | np.number

IntList = Sequence[Int]
RealList = Sequence[Real]
ScalarList = Sequence[Number]

ArrayLike = Sequence[any] | np.ndarray
