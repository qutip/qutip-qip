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


# Functions for gray code decomposition
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

    # this function output is not what's expected because the output of
    # gray_code_gate_info is not what's expected.
    # for this reason, there's no test designed for it, once the error in
    # gray_code_gate_info is corrected, a test will be added
    return(gate_dictionary)


def _sqrt_of_1_qubit_array(input_array):
    """ Finds the square root of a 1 qubit gate for decomposing a multi-qubit
    array into smaller controlled CNOT and 2-qubit two-level unitary.

    # based on Lemma 7.5 of https://arxiv.org/abs/quant-ph/9503016
    """
    # final lemma 7.5 function is not defined - will be dfined if needed

    # This method is for when 1 qubit is not special unitary but the single
    # qubit decomposition functions defined already make a non-special unitary
    # output into a su(2) by the determinant.
    check_gate(Qobj(input_array, dims=[[2] * 1] * 2), 1)
    # method taken
    # from https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix

    U_array = input_array
    tau = U_array[0][0] + U_array[1][1]
    delta = np.linalg.det(U_array)
    s = cmath.sqrt(delta)
    t = cmath.sqrt(tau + 2*s)
    sqrt_U = np.multiply(1/t, np.array([[U_array[0][0]+s, U_array[0][1]],
                                        [U_array[1][0], U_array[1][1]+s]]))
    return(sqrt_U)


def _lemma_6_1(input_array):
    """ Decompose a 3 qubit multi-controlled array.
    https://arxiv.org/abs/quant-ph/9503016
    """
    # add condition to check if input is target for last qubit and rest are
    # controls - easier with a subclass instance
    U = input_array
    check_gate(Qobj(U), num_qubits=1)
    V = _sqrt_of_1_qubit_array(U)
    V_dagger = np.transpose(np.conjugate(V))
    V1_gate = controlled_gate(
        V, controls=[1], target=2, control_value=[1])
    V_dagger_gate = controlled_gate(
        V_dagger, controls=[1], target=2, control_value=[1])
    V2_gate = controlled_gate(
            V, controls=[0], target=2, control_value=[1])
    CNOT_ctrl_0 = Gate("CNOT", controls=0, targets=1)
    return(
        V1_gate,
        CNOT_ctrl_0,
        V_dagger_gate,
        CNOT_ctrl_0,
        V2_gate
        )


def _lemma_6_1_for_4_qubit(input_array):
    """ Method shown on page 17 of https://arxiv.org/abs/quant-ph/9503016
    """
    # generalize both 6.1 functions to when num_qubits < 5

    # add condition to check if input is target for last qubit and rest are
    # controls  - easier with a subclass instance
    U = input_array
    check_gate(Qobj(U), num_qubits=1)
    V = _sqrt_of_1_qubit_array(U)
    # reuse V to get square root of V because V**4 = U
    V = _sqrt_of_1_qubit_array(V)
    V_dagger = np.transpose(np.conjugate(V))
    V1Gate = controlled_gate(
        V, controls=[0], target=[3], control_value=[1])
    V2Gate = controlled_gate(
        V_dagger, controls=[1], target=[3], control_value=[1])
    V3Gate = controlled_gate(
        V, controls=[1], target=[3], control_value=[1])
    V4Gate = controlled_gate(
        V_dagger, controls=[2], target=[3], control_value=[1])
    V5Gate = controlled_gate(
        V, controls=[2], target=[3], control_value=[1])
    CNOT1 = Gate("CNOT", controls=0, targets=1)
    CNOT2 = Gate("CNOT", controls=1, targets=2)
    CNOT3 = Gate("CNOT", controls=0, targets=2)
    return(
        V1Gate,
        CNOT1,
        V2Gate,
        CNOT1,
        V3Gate,
        CNOT2,
        V4Gate,
        CNOT3,
        V5Gate,
        CNOT2,
        V4Gate,
        CNOT3,
        V5Gate
        )


def _lemma_5_1(input_array):
    """ Decompose 2-qubit controlled unitary.
    https://arxiv.org/abs/quant-ph/9503016
    """
    # add condition to check if input is target for last qubit and rest are
    # controls  - easier with a subclass instance
    U = input_array
    check_gate(Qobj(U), num_qubits=1)
    gate_list = decompose_one_qubit_gate(Qobj(U), "ZYZ_PauliX")

    # Change the gate targets for ABC and X gates from default = 0
    # First 2 indices are for A, next 2 are for B and last is for C
    # Note that we have avoided changing the Pauli X indices because it is
    # easier to define a gate with the target qubit.
    target_indices_to_change = [0, 1, 3, 4, 6]
    for i in target_indices_to_change:
        gate_list[i].targets = [1]

    # last is phase whose target is not changed based on Lemma 5.2 of
    # arXiv:quant-ph/9503016v1 but the gate type is changed
    phase_angle = gate_list[-1].arg_value
    phase_gate = Gate(
        "PHASEGATE",
        targets=[0],
        arg_value=phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(phase_angle / np.pi),
    )
    Pauli_X = Gate("X", targets=[1], classical_controls=[0])

    CNOT_ctrl_0 = Gate("CNOT", controls=0, targets=1)
    return (
        gate_list[0],
        gate_list[1],
        Pauli_X,
        CNOT_ctrl_0,
        Pauli_X,
        gate_list[3],
        gate_list[4],
        Pauli_X,
        CNOT_ctrl_0,
        Pauli_X,  # to do : add a condition to check for control value before
        # adding Pauli_X
        gate_list[6],
        phase_gate,
    )


def _control_0_to_pauli_x(input_gate, num_qubits):
    """ If the control value is 0 for qubit_i, this function returns info about
    the indices where pauli X gate should be added.
    """
    two_level_gate_info = _two_level_gate_info(input_gate, num_qubits)
    pauli_x_info = {}
    for i in range(len(two_level_gate_info)):
        control_value_list = two_level_gate_info[i]["control_value ="]
        for j in range(len(control_value_list)):
            if control_value_list[j] == 0:
                pauli_x_partial_info = {}
                gate_indices = [j-1, j]
                pauli_x_partial_info[j] = gate_indices

            pauli_x_info[i] = pauli_x_partial_info

    return(pauli_x_info)


def _multi_cnot_to_two_diff_cnot(input_array, num_qubits):
    """ Decomposes a multicontrolled cnot into a circuit described by multiple
    cnots of smaller controls.

    # based on lemma 7.3 of https://arxiv.org/abs/quant-ph/9503016
    """
    # add condition to check if input is target for last qubit and rest are
    # controls - easier with a subclass instance

    # the gate is not a control on num_qubits-1 qubit
    assert num_qubits >= 5
    cnot_gate_info = {}
    num_qubits_list = list(range(num_qubits))

    m = list(range(2, num_qubits-3))

    # m target gates

    # num_qubits-m-1 target gates






def _decompose_multi_cnot_further():
    """ Decomposes output of lemma 7.3 even further by using only 1 particular
    cnot with different controls.

    # based on lemma 7.2 of https://arxiv.org/abs/quant-ph/9503016
    """


# Added both 7.9 and 7.11 because not sure which one leads to the least number
# of gates.
def _decompose_lemma7_9():
    """ Decomposes a multi-qubit controlled 2x 2 special unitary into a
    decomposition described by CNOT and ABC decomposition gates.


    # based on Lemma 7.9 of https://arxiv.org/abs/quant-ph/9503016
    """




def decompose_general_qubit_gate(input_gate, num_qubits, method):
    """ Decomposes a general qubit gate into description of CNOT and single
    qubit gates.
    """
    check_gate(input_gate, num_qubits)
    if method is None:
        if num_qubits == 1:
            decompose_one_qubit_gate(input_gate, "ZYZ_PauliX")
    else:
        raise MethodError("Choose the decomposition method for num_qubits > 1")
    if method == "two_level":
        two_level_gate_info = _two_level_gate_info(input_gate, num_qubits)
        # make function return two level gates when a gate object has been
        # created - cannot return two-level gate objects at the moment

    else:
        # the code below for num_qubits = 2,3,4 is repitive for now until it's
        # checked that the only difference between the threee methods is just
        # the lemma fn for decomposition

        # once that's verified, a dictionary can be defined with keys for their
        # respective mthods.
        if num_qubits == 2:
            final_gate_dictionary = {}
            two_level_compact_info = _decompose_to_two_level_arrays(
                input_gate, 2, expand=False)
            for i in range(len(two_level_compact_info)):
                input_array = two_level_compact_info[i][1]
                two_level_gate_info = _two_level_gate_info(
                    input_gate, num_qubits)
                total_num_gates = len(two_level_gate_info)
                for j in range(total_num_gates):
                    if len(total_num_gates) == 1:
                        gate_i_info = _lemma_5_1(input_array)
                        final_gate_dictionary[j] = gate_i_info
            # to do : change targets and controls based on two_level_gate_info
                    else:
                        # find position of known decomposition
                        # rest are cnot
                        n = int((len(total_num_gates)+1)/2)
                        # output of _two_level_gate_info is incorrect
                    # this portion is skipped until that output is corrected

        # num_qubits =3, 4 are also expected to have a similar structure as
        # num_qubits =2

        else:
        # for the method when number of qubits is gretaer than or equal to 5
