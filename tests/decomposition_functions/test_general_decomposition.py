

import numpy as np
import pytest

from qutip import (
    Qobj, average_gate_fidelity, rand_unitary
)


from qutip_qip.decompose.decompose_general_qubit_gate import (
    _decompose_to_two_level_arrays, _create_dict_for_two_level_arrays)


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


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
def test_empty_dict_of_two_level_arrays(num_qubits):
    """ Check if empty dictionary is of the same length as the two-level array
    output.
    """
    input_gate = rand_unitary(2**num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False)
    empty_dict_output = _create_dict_for_two_level_arrays(array_decompose)
    assert np.equal(len(empty_dict_output), len(array_decompose))
