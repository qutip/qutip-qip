"""
For the compatibility between qutip-v5 and v4.
"""

from itertools import chain
from functools import reduce
from packaging.version import parse as parse_version
import numpy as np
import qutip


def to_scalar(qobj_or_scalar):
    if isinstance(qobj_or_scalar, qutip.Qobj):
        if qobj_or_scalar.dims == [[1], [1]]:
            return qobj_or_scalar[0, 0]
    return qobj_or_scalar
