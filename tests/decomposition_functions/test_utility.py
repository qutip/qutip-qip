import numpy as np
import pytest

from qutip import Qobj, qeye
from qutip_qip.decompose._utility import (
    check_gate,
    _binary_sequence,
    _gray_code_sequence,
    _gray_code_steps
)


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
    """Checks if correct value is returned or not when the input is not a Qobj
    ."""
    with pytest.raises(TypeError, match="The input matrix is not a Qobj."):
        check_gate(invalid_input, num_qubits=1)


@pytest.mark.parametrize("non_unitary", [Qobj([[1, 1], [0, 1]])])
def test_check_gate_non_unitary(non_unitary):
    """Checks if non-unitary input is correctly identified."""
    with pytest.raises(ValueError, match="Input is not unitary."):
        check_gate(non_unitary, num_qubits=1)


@pytest.mark.parametrize("non_1qubit_unitary", [qeye(4)])
def test_check_gate_non_1qubit(non_1qubit_unitary):
    """Checks if non-unitary input is correctly identified."""
    num_qubits = 1
    with pytest.raises(
         ValueError, match=f"Input is not a unitary on {num_qubits} qubits."):
        check_gate(non_1qubit_unitary, num_qubits)


@pytest.mark.parametrize("unitary", [Qobj([[1, 0], [0, -1]])])
def test_check_gate_unitary_input(unitary):
    """Checks if shape of input is correctly identified."""
    # No error raised if it passes.
    check_gate(unitary, num_qubits=1)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("method", [_gray_code_sequence, _binary_sequence])
def test_full_sequence_length(num_qubits, method):
    """ Tests length of binary and gray code sequence compared to the number of
    qubits.
    """
    expected_len_sequence = 2**num_qubits
    len_from_function = len(method(num_qubits))
    assert np.equal(expected_len_sequence, len_from_function)


def test_binary_sequence():
    """ Checks if the output of binary code function is as expected.
    """
    expected_seq2 = ['00', '01', '10', '11']
    calc_sequence2 = _binary_sequence(2)
    assert expected_seq2 == calc_sequence2

    expected_seq3 = ['000', '001', '010', '011', '100', '101', '110', '111']
    calc_sequence3 = _binary_sequence(3)
    assert expected_seq3 == calc_sequence3


def test_gray_code_sequence():
    """ Checks if output of gray code function is as expected.
    """
    expected_seq2 = ['00', '01', '11', '10']
    calc_sequence2 = _gray_code_sequence(2)
    assert expected_seq2 == calc_sequence2

    expected_seq3 = ['000', '001', '011', '010', '110', '111', '101', '100']
    calc_sequence3 = _gray_code_sequence(3)
    assert expected_seq3 == calc_sequence3


def test_gray_code_steps():
    """ Checks the number of steps and smaller gray code sequence is returned.
    """
    # the expected number of steps is mapping from index1 to index2 and then
    # mapping back to index1 if the first iteration is done in more than 1 steps.

    # for num_qubits = 2, if the mapping is done from '00' to '10'
    expected_num_steps = 5
    expected_gray_code = ['00', '01', '11', '10']
    calc_num_steps_sequence = _gray_code_steps(0, 2, 2)
    calc_num_steps = calc_num_steps_sequence[1]
    calc_gray_code = calc_num_steps_sequence[0]
    assert np.equal(calc_num_steps, expected_num_steps)
    assert expected_gray_code == calc_gray_code

    # for num_qubits = 2, '00' to '01' is only 1 step
    expected_num_steps = 1
    expected_gray_code = ['00', '01']
    calc_num_steps_sequence = _gray_code_steps(0, 1, 2)
    calc_num_steps = calc_num_steps_sequence[1]
    calc_gray_code = calc_num_steps_sequence[0]
    assert np.equal(calc_num_steps, expected_num_steps)
    assert expected_gray_code == calc_gray_code

    # for num_qubits = 3, mapping from '000' to '111'
    expected_num_steps = 9
    expected_gray_code = ['000', '001', '011', '010', '110', '111']
    calc_num_steps_sequence = _gray_code_steps(0, 7, 3)
    calc_num_steps = calc_num_steps_sequence[1]
    calc_gray_code = calc_num_steps_sequence[0]
    assert np.equal(calc_num_steps, expected_num_steps)
    assert expected_gray_code == calc_gray_code
