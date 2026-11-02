import warnings

import scipy.sparse as sp
import numpy as np
from qutip import Qobj

from qutip_qip.operations import ControlledUnitary, Unitary, AngleParametricOp
from qutip_qip.operations.gates import X, SWAP


class GLOBALPHASE(AngleParametricOp):
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

    num_qreg: int = 0
    phi: float
    latex_str: str = r"{\rm GLOBALPHASE}"

    def __post_init__(self) -> None:
        warnings.warn(
            "GLOBALPHASE is deprecated and will be removed in a future version. "
            "Use QubitCircuit.add_global_phase(phase) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__post_init__()

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        r"""
        Return the scalar Qobj corresponding to the global phase factor
        :math:`e^{i \theta}`.
        Returns
        -------
        qobj : qutip.Qobj
            A scalar quantum object with data equal to :math:`e^{i \theta}`.
        """
        return Qobj(np.exp(1.0j * self.phi), dtype=dtype)

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
        N = 1 << num_qubits
        return Qobj(
            np.exp(1.0j * self.phi) * sp.eye(N, N, dtype=complex, format="csr"),
            dims=[[2] * num_qubits, [2] * num_qubits],
            dtype=dtype,
        )

    def inverse(self):
        return GLOBALPHASE(-self.phi)


class TOFFOLI(ControlledUnitary):
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

    num_ctrl_qreg: int = 2
    ctrl_value: int = 0b11

    target: type[Unitary] = X
    latex_str: str = r"{\rm TOFFOLI}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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


class FREDKIN(ControlledUnitary):
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

    num_ctrl_qreg: int = 1
    ctrl_value: int = 1

    target: type[Unitary] = SWAP
    latex_str: str = r"{\rm FREDKIN}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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
