from functools import cache
from typing import Final, Type

import scipy.sparse as sp
import numpy as np
from qutip import Qobj

from qutip_qip.operations import Gate, ControlledGate, AngleParametricGate
from qutip_qip.operations.gates import X, SWAP
from qutip_qip.operations.namespace import NS_GATE


class GLOBALPHASE(AngleParametricGate):
    r"""
    Global phase gate.

    Adds a global phase factor $e^{i \theta}$ to a Quantum Circuit.
    Note: a global phase has no observable physical effect on a system.
    But is using for mathematical correctness for Gate Decompositions.

    Parameters
    ----------
    arg_value : float
        The phase angle $\theta$ to be applied. Passed down to the `ParametricGate`
        base class.
    arg_label : str, optional
        Label for the argument to be shown in the circuit plot.

    Examples
    --------
    >>> from qutip_qip.operations.gates import GLOBALPHASE
    >>> from math import pi
    >>> # Create a global phase gate with an angle of pi
    >>> gate = GLOBALPHASE(pi)
    >>> gate.get_qobj()
    Quantum object: dims=[[1], [1]], shape=(1, 1), type='scalar', dtype=Dense
    Qobj data =
    [[-1.+0.j]]
    >>> gate.get_expanded_qobj(num_qubits=1)
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[-1.  0.]
     [ 0. -1.]]
    """

    __slots__ = ()
    namespace = NS_GATE

    num_qubits: Final[int] = 0
    num_params: Final[int] = 1
    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm GLOBALPHASE}"

    def __repr__(self):
        return f"Gate({self.name}, phase {self.arg_value[0]}) -> Qobj:"

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        r"""
        Return the scalar Qobj corresponding to the global phase factor
        :math:`e^{i \theta}`.
        Returns
        -------
        qobj : qutip.Qobj
            A scalar quantum object with data equal to :math:`e^{i \theta}`.
        """
        phase = self.arg_value[0]
        return Qobj(np.exp(1.0j * phase), dtype=dtype)

    def get_expanded_qobj(self, num_qubits: int, dtype: str = "dense") -> Qobj:
        r"""
        Get the QuTiP quantum object representation expanded for a given number of qubits.

        Constructs the identity matrix $I$ for a system of `num_qubits` and scales
        it by the global phase factor $e^{i \theta}$. The resulting dimension of the
        Hilbert space is $N = 2^{\text{num\_qubits}}$.

        Parameters
        ----------
        num_qubits : int
            The number of qubits over which the global phase is applied.
        dtype : str, default="dense"
            The data type of the Qobj's underlying representation (e.g., "dense").

        Returns
        -------
        qobj : qutip.Qobj
            The expanded unitary matrix representing $e^{i \theta} I$ for the
            $N$-qubit system.
        """
        phase = self.arg_value[0]
        N = 1 << num_qubits
        return Qobj(
            np.exp(1.0j * phase) * sp.eye(N, N, dtype=complex, format="csr"),
            dims=[[2] * num_qubits, [2] * num_qubits],
            dtype=dtype,
        )

    def inverse(self):
        theta = self.arg_value[0]
        return GLOBALPHASE((-theta,))


class TOFFOLI(ControlledGate):
    """
    TOFFOLI gate (CCX).

    Examples
    --------
    >>> from qutip_qip.operations.gates import TOFFOLI
    >>> TOFFOLI.get_qobj()
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
    namespace = NS_GATE

    num_qubits: Final[int] = 3
    num_ctrl_qubits: Final[int] = 2
    ctrl_value: Final[int] = 0b11

    target_gate: Final[Type[Gate]] = X
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm TOFFOLI}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
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
            dtype=dtype,
        )


class FREDKIN(ControlledGate):
    """
    FREDKIN gate (C-SWAP).

    Examples
    --------
    >>> from qutip_qip.operations.gates import FREDKIN
    >>> FREDKIN.get_qobj()
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
    namespace = NS_GATE

    num_qubits: Final[int] = 3
    num_ctrl_qubits: Final[int] = 1
    ctrl_value: Final[int] = 1

    target_gate: Final[Type[Gate]] = SWAP
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm FREDKIN}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
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
            dtype=dtype,
        )
