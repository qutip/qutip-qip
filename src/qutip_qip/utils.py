"""
Module for Helper functions.
"""

from typing import TypeVar, Sequence
from qutip import Qobj

__all__ = [
    "valid_unitary",
    "check_limit",
    "convert_type_input_to_sequence",
]

T = TypeVar("T")  # T can be any type


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


def check_limit(
    input_name: str, input_value: Sequence[T], lower_limit: T, upper_limit: T
):
    if len(input_value) == 0:
        return

    min_element = min(input_value)
    if min_element < lower_limit:
        raise ValueError(
            f"Each entry of {input_name} must be greater than {lower_limit}, but found {min_element}."
        )

    max_element = max(input_value)
    if max_element > upper_limit:
        raise ValueError(
            f"Each entry of {input_name} must be less than {upper_limit}, but found {max_element}."
        )


def convert_type_input_to_sequence(
    input_type: T,
    input_name: str,
    input_value: T | Sequence[T],
) -> Sequence[T]:
    if isinstance(input_value, input_type):
        return [input_value]

    elif isinstance(input_value, Sequence) and not isinstance(input_value, str):
        for i, val in enumerate(input_value):
            if not isinstance(val, input_type):
                raise TypeError(
                    f"All elements in '{input_name}' must be {input_type}. "
                    f"Found {type(val).__name__} ({val}) at index {i}."
                )
        return input_value

    else:
        raise TypeError(
            f"{input_name} must be an {input_type} or sequence of {input_type}, got {input_value}."
        )
