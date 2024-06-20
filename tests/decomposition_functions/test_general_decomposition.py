import numpy as np
import pytest

from qutip import Qobj, average_gate_fidelity, rand_unitary


from qutip_qip.decompose.decompose_general_qubit_gate import (
    _decompose_to_two_level_arrays,
    _create_dict_for_two_level_arrays,
    _partial_gray_code,
    _split_partial_gray_code,
)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6])
def test_two_level_full_output(num_qubits):
    """Check if product of full two level array output is equal to the input."""
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=True
    )

    product_of_U = array_decompose[-1]

    for i in reversed(range(len(array_decompose) - 1)):
        product_of_U = product_of_U
        product_of_U_calculated = np.dot(product_of_U, array_decompose[i])
        product_of_U = product_of_U_calculated

    product_of_U = Qobj(product_of_U, dims=[[2] * num_qubits] * 2)
    fidelity_of_input_output = average_gate_fidelity(product_of_U, input_gate)
    assert np.isclose(fidelity_of_input_output, 1.0)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_empty_dict_of_two_level_arrays(num_qubits):
    """Check if empty dictionary is of the same length as
    the two-level array output.
    """
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    empty_dict_output = _create_dict_for_two_level_arrays(array_decompose)
    assert np.equal(len(empty_dict_output), len(array_decompose))


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_len_partial_grey_code(num_qubits):
    """Check if split gray code output is of the same length
    as the two-level array output.
    """
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    empty_dict_output = _partial_gray_code(num_qubits, array_decompose)
    assert np.equal(len(empty_dict_output), len(array_decompose))


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_keys_partial_grey_code(num_qubits):
    """Check if dictionary keys are consistent in partial grey code.

    The keys are for all two-level gates describing the decomposition and
    are in a reversed order.
    """
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    dict_output = _partial_gray_code(num_qubits, array_decompose)
    gate_key_list = list(
        _partial_gray_code(num_qubits, array_decompose).keys()
    )
    correct_gate_key_list = list(range(1, len(array_decompose) + 1)[::-1])
    assert gate_key_list == correct_gate_key_list


def test_two_qubit_partial_grey_code_output():
    """Checks if the gray code output is as expected."""
    num_qubits = 2
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    func_output = _partial_gray_code(num_qubits, array_decompose)
    expected_output = {
        6: ["11", "10"],
        5: ["01", "11"],
        4: ["01", "11", "10"],
        3: ["00", "01", "11"],
        2: ["00", "01", "11", "10"],
        1: ["00", "01"],
    }
    assert func_output == expected_output


def test_three_qubit_partial_grey_code_output():
    """Checks if the gray code output is as expected."""
    num_qubits = 3
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    func_output = _partial_gray_code(num_qubits, array_decompose)
    expected_output = {
        28: ["110", "111"],
        27: ["111", "101"],
        26: ["110", "111", "101"],
        25: ["111", "101", "100"],
        24: ["110", "111", "101", "100"],
        23: ["101", "100"],
        22: ["011", "010", "110", "111"],
        21: ["011", "010", "110"],
        20: ["011", "010", "110", "111", "101"],
        19: ["011", "010", "110", "111", "101", "100"],
        18: ["010", "110", "111"],
        17: ["010", "110"],
        16: ["010", "110", "111", "101"],
        15: ["010", "110", "111", "101", "100"],
        14: ["011", "010"],
        13: ["001", "011", "010", "110", "111"],
        12: ["001", "011", "010", "110"],
        11: ["001", "011", "010", "110", "111", "101"],
        10: ["001", "011", "010", "110", "111", "101", "100"],
        9: ["001", "011"],
        8: ["001", "011", "010"],
        7: ["000", "001", "011", "010", "110", "111"],
        6: ["000", "001", "011", "010", "110"],
        5: ["000", "001", "011", "010", "110", "111", "101"],
        4: ["000", "001", "011", "010", "110", "111", "101", "100"],
        3: ["000", "001", "011"],
        2: ["000", "001", "011", "010"],
        1: ["000", "001"],
    }
    assert func_output == expected_output


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_len_split_grey_code(num_qubits):
    """Check if length of both dict outputs of split gray code is equal to
    the total number of two-level array gates.

    First dict is made up of n-bit toffoli and second is made up of
    gate that needs decomposition.
    """
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    len_two_level_array = len(array_decompose)
    func_out = _split_partial_gray_code(
        _partial_gray_code(num_qubits, array_decompose)
    )

    # Checks if 2 separate dictionaries are always returned
    assert len(func_out) == 2

    # check length of each dictionary
    assert len(func_out[0]) == len(array_decompose)
    assert len(func_out[1]) == len(array_decompose)


def test_two_qubit_split_partial_grey_code():
    """Checks if output of split partial gray code function is correct
    with expected output for two qubits."""
    num_qubits = 2
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    split_output = _split_partial_gray_code(
        _partial_gray_code(num_qubits, array_decompose)
    )
    expected_toffoli_output = {
        6: None,
        5: None,
        4: ["01"],
        3: ["00"],
        2: ["00", "01"],
        1: None,
    }
    expected_gate_decom_out = {
        6: ["11", "10"],
        5: ["01", "11"],
        4: ["10", "11"],
        3: ["11", "01"],
        2: ["10", "11"],
        1: ["00", "01"],
    }
    assert expected_toffoli_output == split_output[0]
    assert expected_gate_decom_out == split_output[1]


def test_three_qubit_split_partial_grey_code():
    """Checks if output of split partial gray code function is correct
    with expected output for three qubits."""
    num_qubits = 3
    input_gate = rand_unitary(2 ** num_qubits, dims=[[2] * num_qubits] * 2)
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False
    )
    split_output = _split_partial_gray_code(
        _partial_gray_code(num_qubits, array_decompose)
    )
    expected_toffoli_output = {
        28: None,
        27: None,
        26: ["110"],
        25: ["111"],
        24: ["110", "111"],
        23: None,
        22: ["011", "010"],
        21: ["011"],
        20: ["011", "010", "110"],
        19: ["011", "010", "110", "111"],
        18: ["010"],
        17: None,
        16: ["010", "110"],
        15: ["010", "110", "111"],
        14: None,
        13: ["001", "011", "010"],
        12: ["001", "011"],
        11: ["001", "011", "010", "110"],
        10: ["001", "011", "010", "110", "111"],
        9: None,
        8: ["001"],
        7: ["000", "001", "011", "010"],
        6: ["000", "001", "011"],
        5: ["000", "001", "011", "010", "110"],
        4: ["000", "001", "011", "010", "110", "111"],
        3: ["000"],
        2: ["000", "001"],
        1: None,
    }
    expected_gate_decom_out = {
        28: ["110", "111"],
        27: ["111", "101"],
        26: ["101", "111"],
        25: ["100", "101"],
        24: ["100", "101"],
        23: ["101", "100"],
        22: ["111", "110"],
        21: ["110", "010"],
        20: ["101", "111"],
        19: ["100", "101"],
        18: ["111", "110"],
        17: ["010", "110"],
        16: ["101", "111"],
        15: ["100", "101"],
        14: ["011", "010"],
        13: ["111", "110"],
        12: ["110", "010"],
        11: ["101", "111"],
        10: ["100", "101"],
        9: ["001", "011"],
        8: ["010", "011"],
        7: ["111", "110"],
        6: ["110", "010"],
        5: ["101", "111"],
        4: ["100", "101"],
        3: ["011", "001"],
        2: ["010", "011"],
        1: ["000", "001"],
    }
    assert expected_toffoli_output == split_output[0]
    assert expected_gate_decom_out == split_output[1]
