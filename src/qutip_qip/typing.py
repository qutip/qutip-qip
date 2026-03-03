from typing import Sequence, TypeAlias

import numpy as np

__all__ = [
    "Int",
    "Real",
    "Number",
    "IntSequence",
    "RealSequence",
    "ScalarSequence",
    "ArrayLike",
]

# TODO When minimum version is updated to 3.12, use type (PEP 695) in place of TypeAlias

Int: TypeAlias = int | np.integer
Real: TypeAlias = int | float | np.integer | np.floating
Number: TypeAlias = int | float | complex | np.number

IntSequence = Sequence[Int]
RealSequence = Sequence[Real]
ScalarSequence = Sequence[Number]

ArrayLike = Sequence[any] | np.ndarray
