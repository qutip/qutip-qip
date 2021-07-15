import numpy as np
import cmath
import pytest

from qutip import Qobj, average_gate_fidelity, rand_unitary, sigmax, sigmay, sigmaz

from qutip_qip.decompose.single_qubit_gate import (
    _ZYZ_rotation,
    _ZXZ_rotation,
    ABC_decomposition,
    _ZYZ_pauli_X,
    decompose_to_rotation_matrices,
)

from qutip_qip.circuit import decomposed_gates_to_circuit, compute_unitary
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
@pytest.mark.parametrize("method", [_ZYZ_rotation, _ZXZ_rotation, _ZYZ_pauli_X])
def test_single_qubit_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    gate_list = method(gate, target, num_qubits)
    decomposed_gates_circuit = decomposed_gates_to_circuit(gate_list, num_qubits)
    decomposed_gates_final_matrix = compute_unitary(decomposed_gates_circuit)
    fidelity_of_input_output = average_gate_fidelity(
        gate, decomposed_gates_final_matrix
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize("method", ["ZXZ", "ZYZ"])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    gate_list = decompose_to_rotation_matrices(gate, method, target, num_qubits)
    decomposed_gates_circuit = decomposed_gates_to_circuit(gate_list, num_qubits)
    decomposed_gates_final_matrix = compute_unitary(decomposed_gates_circuit)
    fidelity_of_input_output = average_gate_fidelity(
        gate, decomposed_gates_final_matrix
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize("method", [_ZYZ_rotation, _ZXZ_rotation, _ZYZ_pauli_X])
def test_output_is_tuple(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    gate_list = method(gate, target, num_qubits)
    assert isinstance(gate_list, tuple)


# Tests for public functions
@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize("method", ["ZXZ", "ZYZ"])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    gate_list = decompose_to_rotation_matrices(gate, method, num_qubits, target)
    assert isinstance(gate_list, tuple)


@pytest.mark.parametrize(
    "gate", [H, sigmax, sigmay, sigmaz, SQRTNOT, S, T, rand_unitary(2)]
)
@pytest.mark.parametrize("method", ["ZYZ_PauliX"])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    gate_list = ABC_decomposition(gate, method, num_qubits, target)
    assert isinstance(gate_list, tuple)
