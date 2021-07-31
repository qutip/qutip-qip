import numpy as np
import cmath
import pytest

from qutip import (
    Qobj, average_gate_fidelity, rand_unitary
)

from qutip_qip.decompose.decompose_general_qubit_gate import (
                    _decompose_to_two_level_arrays,
                    plot_gray_code_grid
    )


# def test_plot_gray_code_grid():
#    """ Test output type of gray code plotting function.
#    """
# To Do : Use an image comparison decorator


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6, 7])
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


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6, 7])
def test_two_level_compact_output(num_qubits):
    """ Check if product of compact two level array output is equal to the
    input after creating a two-level array.
    """
    input_gate = rand_unitary(2**num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False)

    compact_to_full = []
    for i in range(len(array_decompose)):
        U_two_level = np.identity(2**num_qubits, dtype=complex)
        index_1, index_2 = array_decompose[i][0]
        U_2_qobj = array_decompose[i][1]
        U_2_array = U_2_qobj.full()
        U_two_level[index_1][index_1] = U_2_array[0][0]
        U_two_level[index_2][index_1] = U_2_array[1][0]
        U_two_level[index_1][index_2] = U_2_array[0][1]
        U_two_level[index_2][index_2] = U_2_array[1][1]
        compact_to_full.append(U_two_level)

    product_of_U = compact_to_full[-1]

    for i in reversed(range(len(compact_to_full)-1)):
        product_of_U = product_of_U
        product_of_U_calculated = np.dot(product_of_U, compact_to_full[i])
        product_of_U = product_of_U_calculated

    product_of_U = Qobj(product_of_U, dims=[[2] * num_qubits] * 2)
    fidelity_of_input_output = average_gate_fidelity(
        product_of_U, input_gate
    )
    assert np.isclose(fidelity_of_input_output, 1.0)
