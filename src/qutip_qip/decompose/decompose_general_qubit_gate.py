import numpy as np
import cmath
from qutip import Qobj
from qutip_qip.decompose._utility import (
    check_gate,
    MethodError,
    _gray_code_steps,
    _gray_code_gate_info,
    _paulix_for_control_0,
)
import matplotlib.pyplot as plt
import warnings
from qutip_qip.circuit import Gate
from qutip_qip.operations import controlled_gate
from .decompose_single_qubit_gate import decompose_one_qubit_gate


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


def _two_qubit_lastq_target(last_two_level_array_info):
    """ Finds equivalent circuit of two qubit two-level array when last qubit
    is the target and first is the control with a control value = 1.

    This object is the first array in output from
    _decompose_to_two_level_arrays(input_gate, num_qubits, expand = False)

    Only works for ind1 = 2, ind2 = 3
    """
    last_two_level_qobj = last_two_level_array_info[-1]
    last_two_level_array = last_two_level_qobj.full()

    # check two_level_gate
    check_gate((Qobj(last_two_level_array, dims=[[2] * 2] * 2)), 2)

    # non-trivial indices
    index_list = last_two_level_array_info[0]
    index1, index2 = index_list
    gray_code_dict = _gray_code_gate_info(index1, index2, 2)

    # pauli x gates to make all control 1
    new_gray_code_dict = _paulix_for_control_0(gray_code_dict)
    control_qubit = new_gray_code_dict[0]['controls ='][0]
    target_qubit = new_gray_code_dict[0]['targets ='][0]

    # get ABC decomposition gates
    single_qubit_decomposition = decompose_one_qubit_gate(
        last_two_level_qobj, 'ZYZ_PauliX')
    CNOT_gate = Gate("CNOT", controls=control_qubit, targets=target_qubit)

    # A1, A2 (with default target 0) forming A are at indices 0, 1
    A1 = single_qubit_decomposition[0]
    A2 = single_qubit_decomposition[1]

    # At index 2, pauli x gate with default target 0
    X_gate = single_qubit_decomposition[2]

    # B1, B2 (with default target 0) forming B are at indices 3, 4
    B1 = single_qubit_decomposition[3]
    B2 = single_qubit_decomposition[4]

    # C (with default target 0) is at index 6
    C = single_qubit_decomposition[6]

    # last is the global phase gate which is changed to phasegate
    phase_angle = single_qubit_decomposition[-1].arg_value
    phase_gate = Gate(
        "PHASEGATE",
        targets=[0],
        arg_value=phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(phase_angle / np.pi),
    )

    # change default targets for gates A, B, C
    change_targets = [A1, A2, X_gate, B1, B2, C]
    for i in range(len(change_targets)):
        change_targets[i].targets = [1]

    # form full gate list
    final_gate_list = [
        A1,
        A2,
        X_gate,
        CNOT_gate,
        X_gate,
        B1,
        B2,
        X_gate,
        CNOT_gate,
        X_gate,
        C,
        phase_gate]
    return(final_gate_list)


def _two_qubit_firstq_target(last_two_level_array_info):
    """ Finds equivalent circuit of two qubit two-level array when last qubit
    is the target and first is the control with a control value = 1.

    This object is the first array in output from
    _decompose_to_two_level_arrays(input_gate, num_qubits, expand = False)

    Only works for ind1 = 1, ind2 = 3
    """
    last_two_level_qobj = last_two_level_array_info[-1]
    last_two_level_array = last_two_level_qobj.full()

    # check two_level_gate
    check_gate((Qobj(last_two_level_array, dims=[[2] * 2] * 2)), 2)

    # non-trivial indices
    index_list = last_two_level_array_info[0]
    index1, index2 = index_list
    gray_code_dict = _gray_code_gate_info(index1, index2, 2)

    # pauli x gates to make all control 1
    new_gray_code_dict = _paulix_for_control_0(gray_code_dict)
    control_qubit = new_gray_code_dict[0]['controls ='][0]
    target_qubit = new_gray_code_dict[0]['targets ='][0]

    # get ABC decomposition gates
    single_qubit_decomposition = decompose_one_qubit_gate(
        last_two_level_qobj, 'ZYZ_PauliX')
    CNOT_gate = Gate("CNOT", controls=control_qubit, targets=target_qubit)

    # A1, A2 (with default target 0) forming A are at indices 0, 1
    A1 = single_qubit_decomposition[0]
    A2 = single_qubit_decomposition[1]

    # At index 2, pauli x gate with default target 0
    X_gate = single_qubit_decomposition[2]

    # B1, B2 (with default target 0) forming B are at indices 3, 4
    B1 = single_qubit_decomposition[3]
    B2 = single_qubit_decomposition[4]

    # C (with default target 0) is at index 6
    C = single_qubit_decomposition[6]

    # last is the global phase gate which is changed to phasegate
    phase_angle = single_qubit_decomposition[-1].arg_value
    phase_gate = Gate(
        "PHASEGATE",
        targets=[1],
        arg_value=phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(phase_angle / np.pi),
    )

    # change default targets for gates A, B, C
    change_targets = [A1, A2, X_gate, B1, B2, C]
    for i in range(len(change_targets)):
        change_targets[i].targets = [0]

    # form full gate list
    final_gate_list = [
        A1,
        A2,
        X_gate,
        CNOT_gate,
        X_gate,
        B1,
        B2,
        X_gate,
        CNOT_gate,
        X_gate,
        C,
        phase_gate]
    return(final_gate_list)


def _find_index_for_firstq_target(two_level_array_output, num_qubits):
    """ Find the two-level array index with first qubit as target and rest
    as controls.
    """
    len_two_level = len(two_level_array_output)
    single_step = []

    # create an array of single step in gray code sequence
    for i in range(len_two_level):
        index1, index2 = two_level_array_output[i][0]
        gray_code_info = _gray_code_gate_info(index1, index2, num_qubits)
        if len(gray_code_info) == 1:
            single_step.append(i)

    # find the index with target_qubit = 0
    for i in range(len(single_step)):
        ind = single_step[i]
        index1, index2 = two_level_array_output[ind][0]
        gray_code = _gray_code_gate_info(index1, index2, num_qubits)
        gray_code_target = gray_code[0]['targets =']
        if gray_code_target[0] == 0:
            return(ind)



def decompose_general_qubit_gate(input_gate, num_qubits):
    check_gate(input_gate, num_qubits)
    two_level_array = _decompose_to_two_level_arrays(
        input_gate, num_qubits, expand=False)

    # first object in two level array is a two-level gate with last qubit as
    # target and others as control
    last_target_qubit_array = two_level_array[0]
    # because arrays are returned in a reversed order
    last_target_qubit_array_index = last_target_qubit_array[0]
    if last_target_qubit_array_index != [2**num_qubits-2, 2**num_qubits-1]:
        raise MethodError("The chosen method cannot be used.")

    # look for the two-level matrix with first qubit as target and rest as
    # controls i.e. want target qubit = 0
    ind_for_0_target = _find_index_for_firstq_target(
        two_level_array, num_qubits)
    first_target_qubit_array = two_level_array[ind_for_0_target]
    first_target_qubit_array_index = first_target_qubit_array[0]

    # create a list of all two-level arrays without first or last qubit as
    # targets
    index_list = list(range(len(two_level_array)))
    index_list.remove(0)
    index_list.remove(ind_for_0_target)
    return(index_list)
