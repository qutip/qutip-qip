import numpy as np
import cmath

from qutip import Qobj
from qutip_qip.decompose._utility import (
    check_gate,
    MethodError, _gray_code_steps, gray_code_gate_info
)
import matplotlib.pyplot as plt
import warnings
from qutip_qip.circuit import Gate
from qutip_qip.operations import controlled_gate

from .decompose_single_qubit_gate import decompose_one_qubit_gate

# for unknown labels in two level gates
warnings.filterwarnings("ignore", category=UserWarning)


def plot_gray_code_grid(index_of_state_1, index_of_state_2, num_qubits):
    """ Plots the difference between each step of a gray code sequence.

    .. note::

        If you would prefer to plot the full gray code sequence, specify the
        first and last indices of full sequence. For a n qubit state, the
        indices of interest will be 0 and :math:`n^2-1` respectively.

    In order to create a plot for a 5 qubit two-level unitary with nontrivial
    values 00101 and 10001,

    .. code-block:: python

        plot_gray_code_grid(5,17,5)
        plt.show()

    Parameters
    -----------
    index_of_state_i: int
        The non-trivial indices in a two-level unitary matrix gate.

    num_qubits:
        Number of qubits being acted upon by the quantum gate.

    Returns
    -------
    matplotlib.image.AxesImage
        The preferred plot for mapping from one binary state to another in a
    gray code sequence.
    """
    # create a gray code sequence
    gray_code_and_num_steps = _gray_code_steps(
        index_of_state_1, index_of_state_2, num_qubits)
    gray_code_sequence = gray_code_and_num_steps[0]
    num_steps = gray_code_and_num_steps[-1]

    # create a dictionary for creating an array later
    input_binary_values = {}
    for i in range(num_steps+1):
        bit_array_at_i = []
        a = gray_code_sequence[i]
        for j in range(num_qubits):
            bit_array_at_i.append(int(a[j]))
            input_binary_values[i] = bit_array_at_i

    # array is created for imshow
    input_binary_array = np.full([num_steps+1, num_qubits], None, dtype=float)
    for i in range(num_steps+1):
        input_binary_array[i] = np.array([input_binary_values[i]])

    # size_for_plot = 2*num_qubits  # figsize=(size_for_plot, size_for_plot)
    # figsize could be used to adjust the length of plot based on the size
    # of input

    # information for plot
    fig, ax = plt.subplots(1)
    y_axis = list(range(len(gray_code_sequence)))
    x_axis = list(range(num_qubits))
    ax.set_xticks(x_axis)
    ax.set_yticks(y_axis)
    ax.set_yticklabels(list(reversed(gray_code_sequence)))
    plt.xlabel("Number of Qubits")
    plt.ylabel("Binary")
    plt.title('Zero Code Sequence')
    im = ax.imshow(
        input_binary_array, cmap='binary', extent=[
            x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]], aspect='auto')

    return(im)


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


def _two_level_gate_info(input_gate, num_qubits):
    """ From the output of two level arrays, create a tuple of controlled gate
    information.
    """
    array_decompose = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False)

    full_index_list = []
    for i in range(len(array_decompose)):
        index_list = array_decompose[i][0]
        full_index_list.append(index_list)

    gate_keys = []
    for i in reversed(range(len(array_decompose))):
        gate_string = 'gate'
        gate_string = gate_string + str(i)
        gate_keys.append(gate_string)

    gate_dictionary = {}
    for i in range(len(array_decompose)):
        index_1, index_2 = full_index_list[i]
        gate_i_info = gray_code_gate_info(index_1, index_2, num_qubits)
        gate_key = gate_keys[i]
        gate_dictionary[gate_key] = gate_i_info

    return(gate_dictionary)
