import numpy as np
import pytest
from qutip import Qobj, qeye
from qutip_qip.utils import valid_unitary


# Tests for valid_unitary
@pytest.mark.parametrize(
    "invalid_input",
    [
        np.array([[1, 1], [1, 1]]),
        ([[1, 1], [1, 1]]),
        1.5,
        3,
        (1, 2, 3, 4),
        np.array([[], []]),
        ([[], []]),
        (),
    ],
)
def test_check_gate_non_qobj(invalid_input):
    """Checks if correct value is returned or not when the input is not a Qobj
    ."""
    with pytest.raises(TypeError, match="The input matrix is not a Qobj."):
        valid_unitary(invalid_input, num_qubits=1)


@pytest.mark.parametrize("non_unitary", [Qobj([[1, 1], [0, 1]])])
def test_check_gate_non_unitary(non_unitary):
    """Checks if non-unitary input is correctly identified."""
    with pytest.raises(ValueError, match="Input is not unitary."):
        valid_unitary(non_unitary, num_qubits=1)


@pytest.mark.parametrize("non_1qubit_unitary", [qeye(4)])
def test_check_gate_non_1qubit(non_1qubit_unitary):
    """Checks if non-unitary input is correctly identified."""
    num_qubits = 1
    with pytest.raises(
        ValueError, match=f"Input is not a unitary on {num_qubits} qubits."
    ):
        valid_unitary(non_1qubit_unitary, num_qubits)


@pytest.mark.parametrize("unitary", [Qobj([[1, 0], [0, -1]])])
def test_check_gate_unitary_input(unitary):
    """Checks if shape of input is correctly identified."""
    # No error raised if it passes.
    valid_unitary(unitary, num_qubits=1)
