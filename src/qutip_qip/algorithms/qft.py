"""
This module provides the circuit implementation for Quantum Fourier Transform.
"""

import warnings

import numpy as np
from qutip import Qobj
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Unitary, register_op
from qutip_qip.operations.gates import CPHASE, H, SWAP
from qutip_qip.typing import Int


class QFT(Unitary):
    n: int

    def validate_params(self):
        if not isinstance(self.n, Int) or self.n < 1:
            raise ValueError(f"n must be an integer and >= 1, got {self.n}")

    @property
    def num_qreg(self):
        return self.n

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        N = 1 << self.n
        rows = np.arange(N)
        U = np.exp(2j * np.pi / N * np.outer(rows, rows)) / np.sqrt(N)
        return Qobj(U, dims=[[2] * self.n] * 2, dtype=dtype)


@register_op(QFT)
def _qft(op, bloq):
    for i in range(op.n):
        for j in range(i):
            bloq.add(CPHASE(np.pi / (2 ** (i - j))), (i, j))
        bloq.add(H, i)

    for i in range(op.n // 2):
        bloq.add(SWAP, (i, op.n - 1 - i))


def qft(N: int = 1) -> Qobj:
    """
    Quantum Fourier Transform operator on N qubits.

    Parameters
    ----------
    N : int
        Number of qubits. (default = 1)

    Returns
    -------
    QFT : Qobj
        Quantum Fourier transform operator.

    """
    warnings.warn(
        "qft has been deprecated and will be removed in future. Please use QFT(n).get_qobj() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return QFT(N).get_qobj()


def qft_gate_sequence(
    N: int = 1, swapping: bool = True, to_cnot: bool = False
) -> QubitCircuit:
    """
    Quantum Fourier Transform operator on N qubits returning the gate sequence.

    Parameters
    ----------
    N : int
        Number of qubits.
    swapping : bool, optional
        Flag indicating sequence of swap gates to be applied at the end or
        not (default: True). This is deprecated.
    to_cnot : bool, optional
        Flag to decompose controlled phase gates to CNOT gates
        (default: False). This is deprecated.

    Returns
    -------
    qc : :class:`.QubitCircuit`
        Gate sequence of Hadamard and controlled rotation gates implementing
        QFT.
    """

    warnings.warn(
        "qft_gate_sequence has been deprecated and will be removed in future. Please use QFT instead"
        "Also use of swapping, to_cnot is no longer supported.",
        DeprecationWarning,
        stacklevel=2,
    )

    qc = QubitCircuit(N)
    qc.add(QFT(N), tuple(range(N)))
    return qc
