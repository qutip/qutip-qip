"""
Module for Helper functions.
"""

from qutip import Qobj

__all__ = ["valid_unitary"]


def valid_unitary(gate, num_qubits):
    """Verifies input is a valid quantum gate i.e. unitary Qobj.

    Parameters
    ----------
    gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    num_qubits:
        Total number of qubits in the circuit.
    Raises
    ------
    TypeError
        If the gate is not a Qobj.
    ValueError
        If the gate is not a unitary operator on qubits.
    """
    if not isinstance(gate, Qobj):
        raise TypeError("The input matrix is not a Qobj.")

    if not gate.isunitary:
        raise ValueError("Input is not unitary.")

    if gate.dims != [[2] * num_qubits] * 2:
        raise ValueError(f"Input is not a unitary on {num_qubits} qubits.")
