import numpy as np
import cmath
import pytest

from qutip import (
    Qobj, average_gate_fidelity, rand_unitary, sigmax, sigmay, sigmaz
)
from qutip_qip._decompose.decompose_single_qubit_gate import (
    _ZYZ_rotation,
    _ZXZ_rotation,
    _ZYZ_pauli_X,
)
from qutip_qip._decompose import decompose_one_qubit_gate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import snot, sqrtnot

# Fidelity closer to 1 means the two states are similar to each other
H = snot(1, 0)
sigmax = sigmax()
sigmay = sigmay()
sigmaz = sigmaz()
SQRTNOT = sqrtnot(N=1, target=0)
T = Qobj([[1, 0], [0, cmath.rect(1, np.pi / 4)]])
S = Qobj([[1, 0], [0, 1j]])
target = 0
num_qubits = 1


# Tests for private functions
@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize(
    "method", [_ZYZ_rotation, _ZXZ_rotation, _ZYZ_pauli_X]
)
def test_single_qubit_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    gate_list = method(gate)
    circuit = QubitCircuit(num_qubits)
    circuit.add_gates(gate_list)
    decomposed_gates_final_matrix = circuit.compute_unitary()
    fidelity_of_input_output = average_gate_fidelity(
        gate, decomposed_gates_final_matrix
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize("method", ["ZXZ", "ZYZ", "ZYZ_PauliX"])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    gate_list = decompose_one_qubit_gate(gate, method)
    circuit = QubitCircuit(num_qubits)
    circuit.add_gates(gate_list)
    decomposed_gates_final_matrix = circuit.compute_unitary()
    fidelity_of_input_output = average_gate_fidelity(
        gate, decomposed_gates_final_matrix
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize(
                    "method", [_ZYZ_rotation, _ZXZ_rotation, _ZYZ_pauli_X]
                        )
def test_output_is_tuple(gate, method):
    """Initial matrix and product of final decompositions are same within some
     phase."""
    gate_list = method(gate)
    assert isinstance(gate_list, tuple)


# Tests for public functions
@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize("method", ["ZXZ", "ZYZ", "ZYZ_PauliX"])
def test_check_single_qubit_to_decompose_to_rotations_tuple(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    gate_list = decompose_one_qubit_gate(gate, method)
    assert isinstance(gate_list, tuple)
