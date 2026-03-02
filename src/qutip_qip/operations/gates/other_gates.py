from functools import cache
from typing import Final, Type

import scipy.sparse as sp
import numpy as np
from qutip import Qobj

from qutip_qip.operations import Gate, ControlledGate, AngleParametricGate
from qutip_qip.operations.gates import X, SWAP


class GLOBALPHASE(AngleParametricGate):
    """
    GLOBALPHASE gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import GLOBALPHASE
    """

    __slots__ = "phase"

    num_qubits: Final[int] = 0
    num_params: Final[int] = 1
    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm GLOBALPHASE}"

    def __init__(self, phase: float = 0.0):
        super().__init__(phase)

    def __repr__(self):
        return f"Gate({self.name}, phase {self.arg_value[0]})"

    def get_qobj(self, num_qubits=None):
        phase = self.arg_value[0]
        if num_qubits is None:
            return Qobj(phase)

        N = 2**num_qubits
        return Qobj(
            np.exp(1.0j * phase) * sp.eye(N, N, dtype=complex, format="csr"),
            dims=[[2] * num_qubits, [2] * num_qubits],
        )


class TOFFOLI(ControlledGate):
    """
    TOFFOLI gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import TOFFOLI
    >>> TOFFOLI.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 1. 0.]]
    """

    __slots__ = ()

    num_qubits: Final[int] = 3
    num_ctrl_qubits: Final[int] = 2
    ctrl_value: Final[int] = 0b11

    target_gate: Final[Type[Gate]] = X
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm TOFFOLI}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dims=[[2, 2, 2], [2, 2, 2]],
        )


class FREDKIN(ControlledGate):
    """
    FREDKIN gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import FREDKIN
    >>> FREDKIN.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]]
    """

    __slots__ = ()

    num_qubits: Final[int] = 3
    num_ctrl_qubits: Final[int] = 1
    ctrl_value: Final[int] = 1

    target_gate: Final[Type[Gate]] = SWAP
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm FREDKIN}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dims=[[2, 2, 2], [2, 2, 2]],
        )
