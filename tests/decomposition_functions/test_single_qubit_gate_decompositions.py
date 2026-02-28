import numpy as np
import pytest

from qutip import average_gate_fidelity
from qutip_qip.decompose.decompose_single_qubit_gate import (
    _ZYZ_rotation,
    _ZXZ_rotation,
    _ZYZ_pauli_X,
)
from qutip_qip.decompose import decompose_one_qubit_gate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.std import H, X, Y, Z, S, T, SQRTX

# Fidelity closer to 1 means the two states are similar to each other
target = 0
gate_list = [H, X, Y, Z, SQRTX, S, T]

# TODO Add a custom gate - rand_unitary(2)


# Tests for private functions
@pytest.mark.parametrize("gate", gate_list)
@pytest.mark.parametrize(
    "method", [_ZYZ_rotation, _ZXZ_rotation, _ZYZ_pauli_X]
)
def test_single_qubit_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    gate_list = method(gate.get_qobj())
    circuit = QubitCircuit(1)
    for g in gate_list:
        circuit.add_gate(g, targets=[0])
    decomposed_gates_final_matrix = circuit.compute_unitary()
    fidelity_of_input_output = average_gate_fidelity(
        gate.get_qobj(), decomposed_gates_final_matrix
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize("gate", gate_list)
@pytest.mark.parametrize("method", ["ZXZ", "ZYZ", "ZYZ_PauliX"])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    circuit = QubitCircuit(1)
    gate_list = decompose_one_qubit_gate(gate.get_qobj(), method)
    for g in gate_list:
        circuit.add_gate(g, targets=[0])

    decomposed_gates_final_matrix = circuit.compute_unitary()
    fidelity_of_input_output = average_gate_fidelity(
        gate.get_qobj(), decomposed_gates_final_matrix
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize("gate", gate_list)
@pytest.mark.parametrize(
    "method", [_ZYZ_rotation, _ZXZ_rotation, _ZYZ_pauli_X]
)
def test_output_is_tuple(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    gate_list = method(gate.get_qobj())
    assert isinstance(gate_list, tuple)


# Tests for public functions
@pytest.mark.parametrize("gate", gate_list)
@pytest.mark.parametrize("method", ["ZXZ", "ZYZ", "ZYZ_PauliX"])
def test_check_single_qubit_to_decompose_to_rotations_tuple(gate, method):
    """Initial matrix and product of final decompositions are same within some
    phase."""
    gate_list = decompose_one_qubit_gate(gate.get_qobj(), method)
    assert isinstance(gate_list, tuple)
