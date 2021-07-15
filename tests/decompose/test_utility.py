import numpy as np
import cmath
import pytest

from qutip import Qobj, qeye
from qutip_qip.decompose._utility import check_gate, MethodError, GateError
from qutip_qip.operations import rx, z_gate, t_gate
from qutip_qip.circuit import QubitCircuit, Gate

# Tests for check_gate
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
    """Checks if correct value is returned or not when the input is not a Qobj."""
    with pytest.raises(TypeError, match="The input matrix is not a Qobj."):
        check_gate(invalid_input, num_qubits=1)


@pytest.mark.parametrize("non_unitary", [Qobj([[1, 1], [0, 1]])])
def test_check_gate_non_unitary(non_unitary):
    """Checks if non-unitary input is correctly identified."""
    with pytest.raises(ValueError, match="Input is not unitary."):
        check_gate(non_unitary, num_qubits=1)


@pytest.mark.parametrize("non_qubit_unitary", [qeye(4)])
def test_check_gate_non_unitary(non_qubit_unitary):
    """Checks if non-unitary input is correctly identified."""
    with pytest.raises(ValueError, match="Input is not a unitary on 2 qubits."):
        check_gate(non_qubit_unitary, num_qubits=2)


@pytest.mark.parametrize("unitary", [Qobj([[1, 0], [0, -1]])])
def test_check_gate_unitary_input(unitary):
    """Checks if shape of input is correctly identified."""
    # No error raised if it passes.
    check_gate(unitary, num_qubits=1)
