"""Circuit representation and simulation at the gate level."""

from .circuit import *
from .circuit_latex import *
from .circuitsimulator import *
from ..operations import Gate, Measurement


def _add_deprecation(fun, msg):
    def newfun(*args, **kwargs):
        warnings.warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        return fun(*args, **kwargs)

    return newfun


Gate = _add_deprecation(
    Gate,
    "The class Gate has been moved to qutip_qip.operations."
    "Please use update the import statement.\n",
)
Measurement = _add_deprecation(
    Measurement,
    "The class Measurement has been moved to qutip_qip.operations."
    "Please use update the import statement.\n",
)
