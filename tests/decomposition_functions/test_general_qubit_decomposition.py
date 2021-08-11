import numpy as np
import cmath
import pytest

from qutip import (
    Qobj, average_gate_fidelity, rand_unitary
)
from qutip_qip.circuit import QubitCircuit

from qutip_qip.decompose._utility import _gray_code_gate_info

from qutip_qip.decompose.decompose_general_qubit_gate import (
    _decompose_to_two_level_arrays,
    _two_qubit_lastq_target,
    _two_qubit_firstq_target,
    _find_index_for_firstq_target)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
def test_two_level_full_output(num_qubits):
    """ Check if product of full two level array output is equal to the input.
    """
    input_gate = rand_unitary(2**num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=True)

    product_of_U = array_decompose[-1]

    for i in reversed(range(len(array_decompose)-1)):
        product_of_U = product_of_U
        product_of_U_calculated = np.dot(product_of_U, array_decompose[i])
        product_of_U = product_of_U_calculated

    product_of_U = Qobj(product_of_U, dims=[[2] * num_qubits] * 2)
    fidelity_of_input_output = average_gate_fidelity(
        product_of_U, input_gate
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


def test_two_qubit_lastq_target():
    """ Checks output of two-level gate decomposition when target = 0 and
    control qubit = 1. (ind1 = 2, ind2 = 3)
    """
    input_gate = rand_unitary(2**2, dims=[[2] * 2] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, 2, expand=False)
    lastq_target_ind = array_decompose[0]
    gate_list = _two_qubit_lastq_target(lastq_target_ind)
    qc = QubitCircuit(2, reverse_states=False)
    qc.add_gates(gate_list)
    calc_u = qc.compute_unitary()

    # check fidelity
    array_decompose_full = _decompose_to_two_level_arrays(
        input_gate, 2, expand=True)
    lastq_full_qobj = array_decompose_full[0]
    fidelity_of_input_output = average_gate_fidelity(
        calc_u, lastq_full_qobj
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


def test_two_qubit_firstq_target():
    """ Checks output of two-level gate decomposition when target = 1 and
    control qubit = 0. (ind1 = 1, ind2 = 3)
    """
    input_gate = rand_unitary(2**2, dims=[[2] * 2] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, 2, expand=False)
    lastq_target_ind = array_decompose[1]
    gate_list = _two_qubit_firstq_target(lastq_target_ind)
    qc = QubitCircuit(2, reverse_states=False)
    qc.add_gates(gate_list)
    calc_u = qc.compute_unitary()

    # check fidelity
    array_decompose_full = _decompose_to_two_level_arrays(
        input_gate, 2, expand=True)
    lastq_full_qobj = array_decompose_full[1]
    fidelity_of_input_output = average_gate_fidelity(
        calc_u, lastq_full_qobj
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
def test_find_index_for_firstq_target(num_qubits):
    """ Tests if the index of two-level where first qubit is target is found.
    """
    input_gate = rand_unitary(2**num_qubits, dims=[[2] * num_qubits] * 2)
    two_level_arrays_expand = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False)
    two_level_matrix_position = _find_index_for_firstq_target(
        two_level_arrays_expand, num_qubits)

    # verify target is 0
    ind1, ind2 = two_level_arrays_expand[two_level_matrix_position][0]
    gray_code_sequence_at_pos = _gray_code_gate_info(ind1, ind2, num_qubits)
    gray_code_gate = gray_code_sequence_at_pos[0]
    assert gray_code_gate['targets ='][0] == 0
