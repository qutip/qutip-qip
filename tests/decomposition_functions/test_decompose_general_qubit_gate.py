import numpy as np
import cmath
import pytest

from qutip import (
    Qobj, average_gate_fidelity, rand_unitary
)

from qutip_qip.decompose.decompose_general_qubit_gate import (
                    _decompose_to_two_level_arrays,
                    plot_gray_code_grid,
                    _sqrt_of_1_qubit_array,
    )

from qutip_qip.decompose._utility import gray_code_gate_info

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


def test_sqrt_of_1_qubit_array():
    """ Tests if square root of numpy array is correctly returned.
    """
    num_qubits = 1
    U = rand_unitary(2**num_qubits, dims=[[2] * num_qubits] * 2)
    U_array = U.full()
    sqrtU = _sqrt_of_1_qubit_array(U_array)
    sqrtU_squared = np.matmul(sqrtU, sqrtU)
    fidelity_of_input_output = average_gate_fidelity(
        Qobj(sqrtU_squared, dims=[[2] * 1] * 2), U
    )
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.xfail
def test_gray_code_info():
    """ Tests if the controls, targets are correctly idenitfied.
    """
    # Note the current function is not outputting expected gate control/target
    # info. The shortened gray code sequence should be ['000', '001', '011',
    # '010', '110', '111', '110', '010', '011', '001', '000'] but the gate
    # control and target info should not be returned for '111' to '110'.
    calculated_output3 = gray_code_gate_info(0, 7, 3)
    ex_out = {
     0: {
        'controls =': [0, 1], 'control_value =': ['0', '0'], 'targets =': [2]},
     1: {
        'controls =': [0, 2], 'control_value =': ['0', '1'], 'targets =': [1]},
     2: {
        'controls =': [0, 1], 'control_value =': ['0', '1'], 'targets =': [2]},
     3: {
        'controls =': [1, 2], 'control_value =': ['1', '0'], 'targets =': [0]},
     4: {'controls =': [0, 1], 'control_value =': ['1', '1'], 'targets =': [2]},
     5: {'controls =': [1, 2], 'control_value =': ['1', '0'], 'targets =': [0]},
     6: {
        'controls =': [0, 1], 'control_value =': ['0', '1'], 'targets =': [2]},
     7: {'controls =': [0, 2], 'control_value =': ['0', '1'], 'targets =': [1]},
     8: {'controls =': [0, 1], 'control_value =': ['0', '0'], 'targets =': [2]}
     }
    # currently last gate of the sequence is not output correctly
    assert calculated_output3 == ex_out

    calc_output2 = gray_code_gate_info(0, 2, 2)
    ex_out2 = {
     0: {'controls =': [0], 'control_value =': ['0'], 'targets =': [1]},
     1: {'controls =': [1], 'control_value =': ['1'], 'targets =': [0]},
     2: {'controls =': [0], 'control_value =': ['1'], 'targets =': [1]},
     3: {'controls =': [1], 'control_value =': ['1'], 'targets =': [0]},
     4: {'controls =': [0], 'control_value =': ['0'], 'targets =': [1]},
     }
    assert calc_output2 == ex_out2
