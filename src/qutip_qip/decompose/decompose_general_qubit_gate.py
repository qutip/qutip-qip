import numpy as np
import cmath
from qutip import Qobj
from qutip_qip.decompose._utility import (
    check_gate,
    _binary_sequence,
    _gray_code_sequence,
)


def _decompose_to_two_level_arrays(input_gate, num_qubits, expand=True):
    """Decompose a general qubit gate to two-level arrays.

    Parameters
    -----------
    input_gate : :class:`qutip.Qobj`
        The gate matrix to be decomposed.
    num_qubits : int
        Number of qubits being acted upon by the input_gate
    expand : True
        Default parameter to return the output as full two-level Qobj. If
    `expand = False` then the function returns a tuple of index information
    and a 2 x 2 Qobj for each gate. The Qobj are returned in reversed order.
    """
    check_gate(input_gate, num_qubits)
    input_array = input_gate.full()

    # Calculate the two level numpy arrays
    array_list = []
    index_list = []
    for i in range(2 ** num_qubits):
        for j in range(i + 1, 2 ** num_qubits):
            new_index = [i, j]
            index_list.append(new_index)

    for i in range(len(index_list) - 1):
        index_1, index_2 = index_list[i]

        # Values of single qubit U forming the two level unitary
        a = input_array[index_1][index_1]
        a_star = np.conj(a)
        b = input_array[index_2][index_1]
        b_star = np.conj(b)
        norm_constant = cmath.sqrt(
            np.absolute(a * a_star) + np.absolute(b * b_star)
        )

        # Create identity array and then replace with above values for
        # index_1 and index_2
        U_two_level = np.identity(2 ** num_qubits, dtype=complex)
        U_two_level[index_1][index_1] = a_star / norm_constant
        U_two_level[index_2][index_1] = b / norm_constant
        U_two_level[index_1][index_2] = b_star / norm_constant
        U_two_level[index_2][index_2] = -a / norm_constant

        # Change input by multiplying by above two-level
        input_array = np.dot(U_two_level, input_array)

        # U dagger to calculate the gates
        U__two_level_dagger = np.transpose(np.conjugate(U_two_level))
        array_list.append(U__two_level_dagger)

    # for U6 - multiply input array by U5 and take dagger
    U_last_dagger = input_array
    array_list.append(U_last_dagger)

    if expand is True:
        array_list_with_qobj = []
        for i in reversed(range(len(index_list))):
            U_two_level_array = array_list[i]
            array_list_with_qobj.append(
                Qobj(U_two_level_array, dims=[[2] * num_qubits] * 2)
            )
        return array_list_with_qobj
    else:
        compact_U_information = []
        for i in reversed(range(len(index_list))):
            U_non_trivial = np.full([2, 2], None, dtype=complex)
            index_info = []
            U_index_together = []

            # create index list
            index_1, index_2 = index_list[i]
            index_info = [index_1, index_2]
            U_index_together.append(index_info)

            # create 2 x 2 arrays
            U_two_level = array_list[i]
            U_non_trivial[0][0] = U_two_level[index_1][index_1]
            U_non_trivial[1][0] = U_two_level[index_2][index_1]
            U_non_trivial[0][1] = U_two_level[index_1][index_2]
            U_non_trivial[1][1] = U_two_level[index_2][index_2]
            U_index_together.append(Qobj(U_non_trivial, dims=[[2] * 1] * 2))

            compact_U_information.append(U_index_together)

        return compact_U_information


def _create_dict_for_two_level_arrays(two_level_output):
    """Creates a dictionary with keys for the total number of two-level array
    output. This will be used by other functions to store information about
    SWAP, PauliX gates etc.
    """
    num_two_level_gates = len(two_level_output)

    # create a reversed list of keys based on total number of two-level gates
    # ranging from 1 to n where n is the total number of two-level arrays
    gate_keys = list(range(1, num_two_level_gates + 1))[::-1]
    gate_info_dict = dict.fromkeys(gate_keys)
    return gate_info_dict


def _partial_gray_code(num_qubits, two_level_output):
    """Returns a dictionary of partial gray code sequence for each two-level
    array.

    The input is when output from decomposition array output is non-expanded."""

    # create empty dict
    gate_key_dict = _create_dict_for_two_level_arrays(two_level_output)

    # create a list of non-trivial indices in two level array output
    two_level_indices = []
    for i in range(len(two_level_output)):
        two_level_indices.append(two_level_output[i][0])

    # gray code sequence output as indices of binary sequence and strings
    # respectively
    gray_code_index = _gray_code_sequence(num_qubits, "index_values")
    gray_code_string = _gray_code_sequence(num_qubits)

    # get the partial gray code sequence
    for i in range(len(two_level_indices)):
        partial_gray_code = []
        ind1 = two_level_indices[i][0]
        ind2 = two_level_indices[i][1]

        ind1_pos_in_gray_code = gray_code_index.index(ind1)
        ind2_pos_in_gray_code = gray_code_index.index(ind2)

        if ind1_pos_in_gray_code > ind2_pos_in_gray_code:
            partial_gray_code = [ind2_pos_in_gray_code, ind1_pos_in_gray_code]
        else:
            partial_gray_code = [ind1_pos_in_gray_code, ind2_pos_in_gray_code]

        gate_key_dict[len(two_level_indices) - i] = gray_code_string[
            partial_gray_code[0] : partial_gray_code[1] + 1
        ]

    return gate_key_dict


def _split_partial_gray_code(gate_key_dict):
    """Splits the output of gray code sequence into n-bit Toffoli and
    two-level array gate of interest.

    The output is a list of dictionary of n-bit toffoli and another dictionary
    for the gate needing to be decomposed.

    When the decomposed gates are added to the circuit, n-bit toffoli will be
    used twice - once in the correct order it is and then in a reversed order.

    For cases where there is only 1 step in the gray code sequence, first dictionary
    will be empty and second will need a decomposition scheme.
    """
    n_bit_toffoli_dict = {}
    two_level_of_int = {}
    for key in gate_key_dict.keys():
        if len(gate_key_dict[key]) > 2:
            key_value = gate_key_dict[key]
            two_level_of_int[key] = [key_value[-1], key_value[-2]]
            n_bit_toffoli_dict[key] = key_value[0:-2]
        else:
            two_level_of_int[key] = gate_key_dict[key]
            n_bit_toffoli_dict[key] = None
    output_of_separated_gray_code = [n_bit_toffoli_dict, two_level_of_int]
    return output_of_separated_gray_code
