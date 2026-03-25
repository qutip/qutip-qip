from typing import Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "Int",
    "Real",
    "Number",
    "IntSequence",
    "RealSequence",
    "ScalarSequence",
    "SequenceLike",
]

# TODO When minimum version is updated to 3.12, use type (PEP 695) in place of TypeAlias

Int: TypeAlias = int | np.integer
Real: TypeAlias = int | float | np.integer | np.floating
Number: TypeAlias = int | float | complex | np.number
SequenceLike: TypeAlias = Sequence | np.ndarray

# These are only meant for typecheck, not for isinstance check
# The reason is npt.NDArray is a generic
# In future, one way to overcome this is to to write a TypeGuard or use Pydantic
IntSequence = list[Int] | tuple[Int, ...] | npt.NDArray[np.integer]
RealSequence = (
    list[Real] | tuple[Real, ...] | npt.NDArray[np.floating] | npt.NDArray[np.integer]
)
ScalarSequence = list[Number] | tuple[Number, ...] | npt.NDArray[np.number]
