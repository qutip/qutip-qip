import warnings

from .operations.gates import *

warnings.warn(
    "Importation from qutip_qip.gates is deprecated."
    "Please use e.g.\n from qutip_qip.operations import cnot\n",
    DeprecationWarning,
    stacklevel=2,
)
