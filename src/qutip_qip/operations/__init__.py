"""
Operations on quantum circuits.
"""
import warnings
from .gates import *
from .gateclass import *
from .measurement import *


def _add_deprecation(fun, msg):
    def newfun(*args, **kwargs):
        warnings.warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        return fun(*args, **kwargs)

    return newfun


import qutip_qip.circuit.circuitsimulator as circuitsimulator

gate_sequence_product = _add_deprecation(
    circuitsimulator.gate_sequence_product,
    "The function gate_sequence_product has been moved to qutip_qip.circuit."
    "Please use update the import statement.\n",
)
