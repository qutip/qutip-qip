import numpy as np
import cmath
from qutip import Qobj
from qutip_qip.decompose._utility import (
    check_gate,
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
    for i in range(2**num_qubits):
        for j in range(i+1, 2**num_qubits):
            new_index = [i, j]
            index_list.append(new_index)

    for i in range(len(index_list)-1):
        index_1, index_2 = index_list[i]

        # Values of single qubit U forming the two level unitary
        a = input_array[index_1][index_1]
        a_star = np.conj(a)
        b = input_array[index_2][index_1]
        b_star = np.conj(b)
        norm_constant = cmath.sqrt(
            np.absolute(a*a_star)+np.absolute(b*b_star))

        # Create identity array and then replace with above values for
        # index_1 and index_2
        U_two_level = np.identity(2**num_qubits, dtype=complex)
        U_two_level[index_1][index_1] = a_star/norm_constant
        U_two_level[index_2][index_1] = b/norm_constant
        U_two_level[index_1][index_2] = b_star/norm_constant
        U_two_level[index_2][index_2] = -a/norm_constant

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
            array_list_with_qobj.append(Qobj(
                U_two_level_array, dims=[[2] * num_qubits] * 2))
        return(array_list_with_qobj)
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
            U_index_together.append(
                Qobj(U_non_trivial, dims=[[2] * 1] * 2))

            compact_U_information.append(U_index_together)

        return(compact_U_information)
