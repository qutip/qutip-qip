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


# To do - delete this function after decisinf if it's needed or not
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


# To do - delete this function after decisinf if it's needed or not
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


def _full_CNOT_to_Toffoli(num_qubits):
    """Decomposes a multi-controlled CNOT with last qubit as target into a
    circuit composed of toffoli and ancilla qubits.
    """
    if num_qubits > 3:
        num_ctrl_qubits = num_qubits - 1
        num_ancilla_qubits = num_ctrl_qubits - 1
        # change last qubit due to ancillas
        last_target_qubit = num_ctrl_qubits + num_ancilla_qubits

        # create ctrl qubit list
        ctrl_qubit_list = list(range(num_ctrl_qubits))
        # create ancilla qubit list
        ancilla_qubit_list = list(range(num_ctrl_qubits, last_target_qubit))

        # first toffoli uses first 2 qubits and first ancilla
        Toffoli0 = Gate(
            "TOFFOLI", targets=[ancilla_qubit_list[0]], controls=[0, 1])
        # remove ancilla qubit and ctrl qubits from their respective lists
        ctrl_qubit_list.remove(0)
        ctrl_qubit_list.remove(1)

        # create a list of control and target qubits for toffoli gates
        ctrls_targ_list = []
        for i in range(len(ancilla_qubit_list)-1):
            # index for target qubits
            j = i + 1
            ctrls_at_i = [ctrl_qubit_list[i], ancilla_qubit_list[i]]
            targ_at_i = [ancilla_qubit_list[j]]
            ctrls_targ_list.append([ctrls_at_i, targ_at_i])

        # create list of toffoli gates
        full_toffoli_list = [Toffoli0]
        for i in range(len(ctrls_targ_list)):
            ctrls_i = ctrls_targ_list[i][0]
            targ_i = ctrls_targ_list[i][1]
            toffoli_at_i = Gate("TOFFOLI", targets=targ_i, controls=ctrls_i)
            full_toffoli_list.append(toffoli_at_i)

        # middle CNOT
        CNOT_middle = Gate(
            "CNOT", controls=ancilla_qubit_list[-1], targets=last_target_qubit)

        # full gate list
        full_gate_list = []
        for i in range(len(full_toffoli_list)):
            full_gate_list.append(full_toffoli_list[i])

        full_gate_list.append(CNOT_middle)

        for i in range(len(full_toffoli_list)):
            reversed_list = full_toffoli_list[::-1]
            full_gate_list.append(reversed_list[i])
        return(full_gate_list)


def _total_num_qubits_with_ancilla(num_qubits):
    """Finds total number of qubits when ancilla qubits are added to create a
    qubit circuit.
    """
    if num_qubits > 3:
        num_ctrl_qubits = num_qubits-1
        num_ancilla = num_ctrl_qubits - 1
        total_num_qubits = num_ctrl_qubits + num_ancilla + 1
        return(total_num_qubits)
    else:
        return(num_qubits)


def _two_level_CNOT_to_Toffoli(num_qubits):
    """When a 2-level unitary with last target is supposed to be decomposed,
    4 multi-controlled CNOT will act on num_qubits-1 qubits.
    """
    return(_full_CNOT_to_Toffoli(num_qubits-1))


def _sqrt_of_1_qubit_array(input_array):
    """ Used for decomposition of 3-qubit gate.

    Finds the square root of a 1 qubit gate for decomposing a multi-qubit
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


def _two_qubit_last_target(last_two_level_array, target_qubit, ctrl_qubit):
    """ Finds equivalent circuit of two qubit two-level array when last qubit
    is the target and second-last qubit is the control with a
    control value = 1.

    This object is the first array in output from
    _decompose_to_two_level_arrays(input_gate, num_qubits, expand = False)"""
    last_two_level_qobj = Qobj(last_two_level_array, dims=[[2] * 1] * 2)

    # check two_level_gate
    check_gate(last_two_level_qobj, 1)

    # decompose the non-trivial single qubit gate matrix
    single_qubit_decomposition = decompose_one_qubit_gate(
        last_two_level_qobj, 'ZYZ_PauliX')
    CNOT_gate = Gate("CNOT", controls=ctrl_qubit, targets=target_qubit)

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
        targets=[ctrl_qubit],
        arg_value=phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(phase_angle / np.pi),
    )

    # change default targets for gates A, B, C
    change_targets = [A1, A2, X_gate, B1, B2, C]
    for i in range(len(change_targets)):
        change_targets[i].targets = [target_qubit]

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


def _threeq_last_target(last_two_level_array):
    """Defines a method to decompose a 3-qubit two-level array with last
    qubit as target and first 2 as controls.

    This method was added because the current general method decomposes CNOT's
    upto Toffoli gates.

    # follows Lemma 6.1 of arXiv:quant-ph/9503016v1
    """
    V = _sqrt_of_1_qubit_array(last_two_level_array)

    V_dagger = V.conj().T

    full_gate_list = []

    # first set of gates with 1 as ctrl and 2 as target
    V12_gate_list = _two_qubit_last_target(V, 2, 1)
    for i in range(len(V12_gate_list)):
        full_gate_list.append(V12_gate_list[i])

    # CNOT between V and V_dagger
    full_gate_list.append(Gate("CNOT", controls=0, targets=1))

    # gate list from V_dagger with 1 as ctrl and 2 as target
    V_dagger_12_gate_list = _two_qubit_last_target(V_dagger, 2, 1)
    for i in range(len(V_dagger_12_gate_list)):
        full_gate_list.append(V_dagger_12_gate_list[i])

    # CNOT between V_dagger and V
    full_gate_list.append(Gate("CNOT", controls=0, targets=1))

    # last set of gates with 0 as ctrl and 2 as target
    V02_gate_list = _two_qubit_last_target(V, 2, 0)
    for i in range(len(V02_gate_list)):
        full_gate_list.append(V02_gate_list[i])

    return(full_gate_list)


def _threeq_first_target(last_two_level_array):
    """Defines a method to decompose a 3-qubit two-level array with first
    qubit as target and other 2 as controls.

    This method was added because the current general method decomposes CNOT's
    upto Toffoli gates.

    # follows Lemma 6.1 of arXiv:quant-ph/9503016v1
    """
    V = _sqrt_of_1_qubit_array(last_two_level_array)

    V_dagger = V.conj().T

    full_gate_list = [Gate("X", targets=[2])]

    # first set of gates with 1 as ctrl and 2 as target
    V12_gate_list = _two_qubit_last_target(V, 0, 1)
    for i in range(len(V12_gate_list)):
        full_gate_list.append(V12_gate_list[i])

    # CNOT between V and V_dagger
    full_gate_list.append(Gate("CNOT", controls=2, targets=1))

    # gate list from V_dagger with 1 as ctrl and 2 as target
    V_dagger_12_gate_list = _two_qubit_last_target(V_dagger, 0, 1)
    for i in range(len(V_dagger_12_gate_list)):
        full_gate_list.append(V_dagger_12_gate_list[i])

    # CNOT between V_dagger and V
    full_gate_list.append(Gate("CNOT", controls=2, targets=1))

    # last set of gates with 0 as ctrl and 2 as target
    V02_gate_list = _two_qubit_last_target(V, 0, 2)
    for i in range(len(V02_gate_list)):
        full_gate_list.append(V02_gate_list[i])
    full_gate_list.append(Gate("X", targets=[2]))
    return(full_gate_list)


def _two_q_CNOT_gates(two_qubit_gate):
    """Create full list of Pauli X and CNOT gates for final decomposition.
    """
    two_level_info = _decompose_to_two_level_arrays(
        two_qubit_gate, 2, expand=False)

    # create a list of non-trivial indices
    non_trivial_indices = []
    for i in range(len(two_level_info)):
        non_trivial_indices.append(two_level_info[i][0])

    # gate dict for two-level gates
    gate_keys = list(range(len(two_level_info)))
    full_gate_dict = {}

    # gray code steps
    full_gray_code_info = []
    for i in range(len(two_level_info)):
        ind1, ind2 = non_trivial_indices[i]
        full_gray_code_info.append(_gray_code_gate_info(ind1, ind2, 2))

    # add pauli gates
    gray_code_ctrl_val_1 = {}
    for i in range(len(full_gray_code_info)):
        gray_pauli_i = _paulix_for_control_0(full_gray_code_info[i])
        gray_code_ctrl_val_1[i] = gray_pauli_i

    out = gray_code_ctrl_val_1

    # create a dictionary of flattened lists
    full_dict = {}
    for i in range(6):
        for_two_level_i = out[i]
        if isinstance(for_two_level_i, dict):
            full_dict[i] = [*out[i].values()]
        elif isinstance(for_two_level_i, list):
            if len(for_two_level_i) > 2:
                full_dict[i] = for_two_level_i
            elif len((for_two_level_i)) == 2:
                forwa_seq = for_two_level_i[0]
                back_seq = for_two_level_i[1]
                forw_list = []
                back_list = []
                for_key_list = [*forwa_seq.keys()]
                back_key_list = [*back_seq.keys()]
                for j in for_key_list:
                    if isinstance(forwa_seq[j], list):
                        for k in range(len(forwa_seq[j])):
                            forw_list.append(forwa_seq[j][k])
                    else:
                        forw_list.append(forwa_seq[j])
                for p in back_key_list:
                    if isinstance(back_seq[p], list):
                        for q in range(len(back_seq[p])):
                            back_list.append(back_seq[p][q])
                    else:
                        back_list.append(back_seq[p])
                full_dict[i] = [forw_list, back_list]

    # create a list of keys that will need CNOT gates or decomposition
    cnot_ind = []
    for i in range(6):
        if len(full_dict[i]) == 2:
            cnot_ind.append(i)
    forw_ind_list = {}
    backw_ind_list = {}
    full_ind_list = []
    for i in cnot_ind:
        gray_code_info = full_dict[i]
        forw = gray_code_info[0]
        backw = gray_code_info[1]
        forw_ind_i = []
        back_ind_i = []
        for j in forw:
            if isinstance(j, dict):
                ind_list = [x for x, q in enumerate(forw) if q == j]
                forw_ind_i.extend(ind_list)

            full_ind2_list = set()

            for l in range(len(forw_ind_i)):
                full_ind2_list.add(forw_ind_i[l])

        forw_ind_list[i] = list(full_ind2_list)

        for j in backw:
            if isinstance(j, dict):
                ind_list = [x for x, q in enumerate(backw) if q == j]
                back_ind_i.extend(ind_list)
            full_ind2_list = set()

            for l in range(len(back_ind_i)):
                full_ind2_list.add(back_ind_i[l])

        backw_ind_list[i] = list(full_ind2_list)

    # create dictionary for those that need to use the decomposition function
    # from forward info
    dec_gate_ind_dict = {}
    for i in cnot_ind:
        ind_at_k = forw_ind_list[i][-1]
        dec_gate_ind_dict[i] = ind_at_k
        forw_ind_list[i].remove(ind_at_k)

    # add other keys with only 1 step in the gray code
    for i in range(6):
        if len(full_dict[i]) == 1:
            dec_gate_ind_dict[i] = 0
        elif len(full_dict[i]) > 2:
            for j in range(len(full_dict[i])):
                if isinstance(full_dict[i][j], dict):
                    dec_gate_ind_dict[i] = j

    # sort the entire decomposition dictionary
    dec_gate_ind_dict_sorted = {}
    for i in range(6):
        dec_gate_ind_dict_sorted[i] = dec_gate_ind_dict[i]

    # create full dictionary with CNOT gates and insert in main full dict
    for i in cnot_ind:
        full_gray_code_for_two_level_gate_i = full_dict[i]
        forw_gray_code = full_gray_code_for_two_level_gate_i[0]
        gate_list = []
        for j in forw_ind_list[i]:
            ind_for_insert_cnot = j
            gate_ctrl_targets = forw_gray_code[j]
            ctrl_qubit = gate_ctrl_targets['controls ='][0]
            targ_qubit = gate_ctrl_targets['targets ='][0]
            full_dict[i][0][j] = Gate(
                "CNOT", controls=ctrl_qubit, targets=targ_qubit)
            # can skip extra step for backward gray code because except
            # for the gates that need decomposition function, going
            # forward and backward should be same gate info
            full_dict[i][1] = Gate(
                "CNOT", controls=ctrl_qubit, targets=targ_qubit)

    # now decompose the 1 qubit non-trivial arrays
    for i in range(6):
        input_array = two_level_info[i][1].full()
        full_gray_code_for_two_level_gate_i = full_dict[i]
        if len(full_gray_code_for_two_level_gate_i) > 2:
            full_gray_code = full_gray_code_for_two_level_gate_i
            ind_to_insert_gates = dec_gate_ind_dict_sorted[i]
            ctrl_targ_info = full_gray_code[ind_to_insert_gates]
            ctrl_qubit = ctrl_targ_info['controls ='][0]
            targ_qubit = ctrl_targ_info['targets ='][0]
            gate_list = _two_qubit_last_target(
                input_array, targ_qubit, ctrl_qubit)
            full_gray_code[ind_to_insert_gates] = gate_list
            full_dict[i] = full_gray_code
        elif len(full_gray_code_for_two_level_gate_i) == 1:
            forw_gray_code = full_gray_code_for_two_level_gate_i[0]
            ind_to_insert_gates = dec_gate_ind_dict_sorted[i]
            # in forward gray code
            if isinstance(forw_gray_code, dict):
                ctrl_qubit = forw_gray_code['controls ='][0]
                targ_qubit = forw_gray_code['targets ='][0]
                gate_list = _two_qubit_last_target(
                    input_array, targ_qubit, ctrl_qubit)
                full_dict[i] = gate_list
        else:
            forw_gray_code = full_gray_code_for_two_level_gate_i[0]
            ind_to_insert_gates = dec_gate_ind_dict_sorted[i]
            # in forward gray code
            ctrl_targ_info = forw_gray_code[ind_to_insert_gates]
            ctrl_qubit = ctrl_targ_info['controls ='][0]
            targ_qubit = ctrl_targ_info['targets ='][0]
            gate_list = _two_qubit_last_target(
                input_array, targ_qubit, ctrl_qubit)
            # forw_gray_code[ind_to_insert_gates] = gate_list
            full_dict[i][0][ind_to_insert_gates] = gate_list
    return(full_dict)


def _CNOT_gate_to_last_target(step_dict_at_i):
    """ If the two-level unitary does not describe a unitary with the target as
    first/last target then CNOT gates are needed to get to extreme target
    two-level unitary to be able to use known decomposition scheme.
    """
    # this could be done via SWAP and resolve_gates as well
    if len(step_dict_at_i) == 1:
        gate_info = step_dict_at_i[0]

        num_qubits = len(gate_info['controls ='])+1

        if gate_info['targets ='][0] != num_qubits-1:
            ctrl_qubit = num_qubits-1
            target_qubit = gate_info['targets ='][0]
            CNOT1 = Gate("CNOT", controls=target_qubit, targets=ctrl_qubit)
            CNOT2 = Gate("CNOT", controls=ctrl_qubit, targets=target_qubit)
            CNOT3 = Gate("CNOT", controls=target_qubit, targets=ctrl_qubit)
            gate_list = [CNOT1, CNOT2, CNOT3]
            return(gate_list)
        else:
            return(step_dict_at_i)


def decompose_general_qubit_gate(input_gate, num_qubits):
    """ Returns a dictionary or a list of gates forming input gate and number
    of qubits in the circuit describing the decomposition.
    """

    check_gate(input_gate, num_qubits)

    if num_qubits == 1:
        # To do - add choice of single qubit scheme
        return(decompose_one_qubit_gate(input_gate, "ZYZ"), num_qubits)
    elif num_qubits == 2:
        return(_two_q_CNOT_gates(input_gate), num_qubits)
    else:
        two_level_array = _decompose_to_two_level_arrays(
            input_gate, num_qubits, expand=False)

        # first object in two level array is a two-level gate with last qubit
        # as target and others as control
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
