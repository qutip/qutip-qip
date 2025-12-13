from collections.abc import Iterable
import numbers
import numpy as np


def _to_array(params, num_qubits):
    """
    Transfer a parameter to an array.
    """
    if isinstance(params, numbers.Real):
        return np.asarray([params] * num_qubits)
    elif isinstance(params, Iterable):
        return np.asarray(params)
