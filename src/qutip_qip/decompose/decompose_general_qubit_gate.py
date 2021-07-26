import numpy as np
import cmath

from qutip import Qobj
from qutip_qip.decompose._utility import (
    check_gate,
    MethodError,
)

import warnings
from qutip_qip.circuit import Gate
from qutip_qip.operations import controlled_gate

from .decompose_single_qubit_gate import decompose_one_qubit_gate

# for unknown labels in two level gates
warnings.filterwarnings("ignore", category=UserWarning)


def _decompose_to_two_level_arrays(input_gate, num_qubits):
    """Decompose a general qubit gate to two-level arrays.
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

    # index_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
    for i in range(len(index_list)-1):
        index_1, index_2 = index_list[i]

        # Values of single qubit U forming the two level unitary
        a = input_array[index_1][index_1]
        a_star = np.conj(a)
        b = input_array[index_2][index_1]
        b_star = np.conj(b)
        norm_constant = cmath.sqrt(np.absolute(a*a_star)+np.absolute(b*b_star))

        # Create identity array and then replace with above values for index_1
        # and index_2
        U_two_level = np.identity(2**num_qubits, dtype=complex)
        U_two_level[index_1][index_1] = a_star/norm_constant
        U_two_level[index_2][index_1] = b/norm_constant
        U_two_level[index_1][index_2] = b_star/norm_constant
        U_two_level[index_2][index_2] = -a/norm_constant

        # Change input by multiplying by above two-level
        input_array = np.dot(U_two_level, input_array)

        # U dagger to calculate the gates
        U__two_level_dagger = np.transpose(np.conjugate(U_two_level))
        U__two_level_dagger = Qobj(U__two_level_dagger, dims=[[2] * num_qubits] * 2)
        array_list.append(U__two_level_dagger)

    # for U6 - multiply input array by U5 and take dagger
    U_last_dagger = input_array
    array_list.append(Qobj(U_last_dagger, dims=[[2] * num_qubits] * 2))
    return(array_list)
