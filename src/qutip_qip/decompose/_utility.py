from qutip import Qobj
import numpy as np
import copy
from qutip_qip.circuit import Gate


class MethodError(Exception):
    """When invalid method is chosen, this error is raised."""

    pass


def check_gate(gate, num_qubits):
    """Verifies input is a valid quantum gate.

    Parameters
    ----------
    gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    num_qubits:
        Total number of qubits in the circuit.
    Raises
    ------
    TypeError
        If the gate is not a Qobj.
    ValueError
        If the gate is not a unitary operator on qubits.
    """
    if not isinstance(gate, Qobj):
        raise TypeError("The input matrix is not a Qobj.")
    if not gate.check_isunitary():
        raise ValueError("Input is not unitary.")
    if gate.dims != [[2] * num_qubits] * 2:
        raise ValueError(f"Input is not a unitary on {num_qubits} qubits.")


def _binary_sequence(num_qubits):
    """ Defines the binary sequence list for basis vectors of a n-qubit gate.
    The string at index `i` is also the row/column index for a basis vector in
    a numpy array.
    """
    old_sequence = ['0', '1']
    full_binary_sequence = []

    if num_qubits == 1:
        full_binary_sequence = old_sequence
    else:
        for x in range(num_qubits-1):
            full_binary_sequence = []
            zero_append_sequence = ['0' + x for x in old_sequence]
            full_binary_sequence.extend(zero_append_sequence)
            one_append_sequence = ['1' + x for x in old_sequence]
            full_binary_sequence.extend(one_append_sequence)
            old_sequence = full_binary_sequence

    return(full_binary_sequence)


def _gray_code_sequence(num_qubits, output_form=None):
    """ Finds the sequence of gray codes for basis vectors by using logical
    XOR operator of Python.
    For print( _gray_code_sequence(2)), the output is [0, 1, 3, 2] i.e. in
    terms of the binary sequence, the output is ['00', '01', '11', '10'].
    https://docs.python.org/3/library/operator.html#operator.xor'
    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit
    output_form : :"index_values" or None
        The format of output list. If a string "index_values" is provided then
    the function's output is in terms of array indices of the binary sequence.
    The default is a list of binary strings.
    Returns
    --------
    list
        List of the gray code sequence in terms of array indices or binary
        sequence positions.
    """
    gray_code_sequence_as_array_indices = []

    for x in range(2**num_qubits):
        gray_code_at_x = x ^ x // 2  # floor operator to shift bits by 1
        # when the shift is done, the new spot is filled with a new value.
        gray_code_sequence_as_array_indices.append(gray_code_at_x)
        if output_form == "index_values":
            output = gray_code_sequence_as_array_indices
        else:
            gray_code_as_binary = []
            binary_sequence_list = _binary_sequence(num_qubits)
            for i in gray_code_sequence_as_array_indices:
                gray_code_as_binary.append(binary_sequence_list[i])
            output = gray_code_as_binary
    return(output)


def _gray_code_steps(index_of_state_1, index_of_state_2, num_qubits):
    """ Finds the sequence mapping from state 1 to state 2.
    State 1 and 2 define the basis vectors of non-trivial values in a two-level
    unitary matrix. Here, the inputs are their respective indices in a quantum
    gate's array.
    This function finds the number of steps between both states when only 1 bit
    can be changed at each step. In addition, a partial sequence going from
    basis state 1 to state 2 and another partial sequence mapping back from
    state 2 to state 1. The last gray sequence value in forward mapping is used
    to define the two-level unitary.
    """
    # check array values are not the same
    try:
        assert (index_of_state_1 != index_of_state_2)
    except AssertionError:
        raise IndexError("Both indices need to be different.")

    # array index for original binary sequence
    indices_of_array = []
    for i in range(2**num_qubits):
        indices_of_array.append(i)

    # Check both indices could be indices for num_qubits array
    try:
        assert all(x in indices_of_array for x in [
                index_of_state_1, index_of_state_2])
    except AssertionError:
        raise IndexError(
            "At least one of the input indices is invalid.")

    # get the basis vector strings
    binary_sequence_list = _binary_sequence(num_qubits)
    state_1_binary = binary_sequence_list[index_of_state_1]
    state_2_binary = binary_sequence_list[index_of_state_2]

    # compare binary sequence positions to gray code sequence
    # finds number of steps neeeded to go from one state to another
    gray_code_sequence = _gray_code_sequence(num_qubits)
    state_1_gray_code_index = gray_code_sequence.index(state_1_binary)
    state_2_gray_code_index = gray_code_sequence.index(state_2_binary)
    num_steps_gray_code = np.abs(
        state_2_gray_code_index - state_1_gray_code_index)

    if num_steps_gray_code == 1:
        num_steps_gray_code = num_steps_gray_code
    else:  # repeat the mapping back to initial index
        num_steps_gray_code = 2*num_steps_gray_code - 1
    # create a smaller gray code sequence between the two states of interest
    if state_1_gray_code_index < state_2_gray_code_index:
        gray_code = _gray_code_sequence(num_qubits)[
            state_1_gray_code_index:state_2_gray_code_index+1]
    else:  # check math for reversed order
        # what is the difference between using gates from left to right vs
        # right to left - with current code, the order is mixed.
        gray_code = _gray_code_sequence(num_qubits)[
            state_2_gray_code_index:state_1_gray_code_index+1]
    gray_code_forward = gray_code
    gray_code_backward = gray_code_forward[0:len(gray_code_forward)-1][::-1]

    if num_steps_gray_code == 1:
        return(gray_code_forward, num_steps_gray_code)
    else:
        return(gray_code_forward, gray_code_backward, num_steps_gray_code)


def _gray_code_gate_info(index_of_state_1, index_of_state_2, num_qubits):
    """ Finds information about control and targets for CNOT gate in a gray
    code sequence.

    Returns
    -------
        A list of two dictionaries containing information about gate controls
    and targets.

    `forward_step_iteration_dictionary` contains information
    iterating over the gray code sequence from state 1 to state 2. Last item
    will contain information about the two-level unitary gate.

    `backward_step_iteration_dictionary` contains information mapping back to
    initial state after the two-level unitary circuit has been added to the
    circuit.
    """
    gray_code_info = _gray_code_steps(
        index_of_state_1, index_of_state_2, num_qubits)
    num_steps = gray_code_info[-1]
    if num_steps > 1:
        gray_code_forward = gray_code_info[0]
        gray_code_backward = gray_code_info[1]
    else:
        gray_code_forward = gray_code_info[0]

    # repeat the mapping from last index to first initial index
    if num_steps == 1:
        gray_code = gray_code_forward

        # make a dictionary of an array of binary values
        # find position of 1's
        input_binary_values = {}

        for i in range(num_steps+1):
            bit_array_at_i = []
            a = gray_code[i]
            for j in range(num_qubits):
                bit_array_at_i.append(a[j])

            input_binary_values[i] = bit_array_at_i

        # compare the array values
        step_iteration_dictionary = {}
        for i in range(num_steps):
            a = input_binary_values[i]
            b = input_binary_values[i+1]
            controls = []
            control_value = []
            target = []
            all_together = {}
            for j in range(num_qubits):
                if a[j] == b[j]:
                    controls.append(j)
                    control_value.append(a[j])
                else:
                    target.append(j)
                all_together['controls ='] = controls
                all_together['control_value ='] = control_value
                all_together['targets ='] = target
            step_iteration_dictionary[i] = all_together

        return(step_iteration_dictionary)
    else:
        # make a dictionary of an array of binary values
        # find position of 1's
        forward_input_binary_values = {}
        for i in range(len(gray_code_forward)):
            bit_array_at_i = []
            a = gray_code_forward[i]
            for j in range(num_qubits):
                bit_array_at_i.append(a[j])

            forward_input_binary_values[i] = bit_array_at_i

        backward_input_binary_values = {}
        for i in range(len(gray_code_backward)):
            bit_array_at_i = []
            a = gray_code_backward[i]
            for j in range(num_qubits):
                bit_array_at_i.append(a[j])

            backward_input_binary_values[i] = bit_array_at_i

        # compare the array values
        step_iteration_dictionary = []
        forward_step_iteration_dictionary = {}
        backward_step_iteration_dictionary = {}

        # gate information going forward
        for i in range(len(gray_code_forward)-1):
            a = forward_input_binary_values[i]
            b = forward_input_binary_values[i+1]
            controls = []
            control_value = []
            target = []
            all_together = {}
            for j in range(num_qubits):
                if a[j] == b[j]:
                    controls.append(j)
                    control_value.append(a[j])
                else:
                    target.append(j)
                all_together['controls ='] = controls
                all_together['control_value ='] = control_value
                all_together['targets ='] = target
            forward_step_iteration_dictionary[i] = all_together
        step_iteration_dictionary.append(
            copy.deepcopy(forward_step_iteration_dictionary))
        # gate information going back to initial basis state
        for i in range(len(gray_code_backward)-1):
            a = backward_input_binary_values[i]
            b = backward_input_binary_values[i+1]
            controls = []
            control_value = []
            target = []
            all_together = {}
            for j in range(num_qubits):
                if a[j] == b[j]:
                    controls.append(j)
                    control_value.append(a[j])
                else:
                    target.append(j)
                all_together['controls ='] = controls
                all_together['control_value ='] = control_value
                all_together['targets ='] = target
            backward_step_iteration_dictionary[
                int((num_steps+1)/2)+i] = all_together

        step_iteration_dictionary.append(
            copy.deepcopy(backward_step_iteration_dictionary))
        return(step_iteration_dictionary)


def _paulix_for_control_0(step_iteration_dictionary):
    """ Returns gray code info adjusted for control value = 0. Wherever the
    control value was changed, Pauli X gate was also added.
    """
    n = len(step_iteration_dictionary)
    # when there's only one step going from state 1 to state 2
    if n == 1:
        # find array indices with control value 0
        gray_code_info = step_iteration_dictionary
        sub_dict = gray_code_info[0]
        array_gate_info_with_control_0 = []
        num_control_qubits = len(sub_dict['control_value ='])
        new_target_info = []
        target_info = sub_dict['controls =']
        ctrl_value = sub_dict['control_value =']
        for j, k in enumerate(ctrl_value):
            if k == '0':
                new_target = target_info[j]
                new_target_info.append(new_target)
                # change control value to be 1 if it is 0
                gray_code_info[0]['control_value ='] = ['1']*num_control_qubits
            # else:
            #    return(gray_code_info)
        if len(new_target_info) != 0:
            # returns info about which two-level array gate decomposition
            # needs pauli x to correct ctrl_value = 0
            array_gate_info_with_control_0.append([0, new_target_info])
        else:
            return(gray_code_info)

        # create array of pauli gates to put before and after the decomposition
        # info from gray code
        pauli_gate_array_forward = []
        gate_targets_from_ctrl_0 = array_gate_info_with_control_0[0][1]
        for i in range(len(gate_targets_from_ctrl_0)):
            target_qubit = gate_targets_from_ctrl_0[i]
            paulix = Gate("X", targets=target_qubit)
            pauli_gate_array_forward.append(paulix)

        pauli_gate_array_backward = pauli_gate_array_forward[::-1]
        # create the new gate array with pauli gates included
        full_gate_array = []
        full_gate_array.extend(pauli_gate_array_forward)
        full_gate_array.append(gray_code_info[0])
        # there's only 1 gate i.e. 1 step
        full_gate_array.extend(pauli_gate_array_backward)

        # what to return
        if len(full_gate_array) == 0:
            return(gray_code_info)
        else:
            return(full_gate_array)
    else:
        # gray code info
        full_gray_code_info = step_iteration_dictionary
        forward_gray_code = full_gray_code_info[0]
        backward_gray_code = full_gray_code_info[1]

        # pauli x from forward gray code
        for_array_gate_info_with_control_0 = []
        gate_keys = []  # the gates that need pauli x for ctrl value =0

        for i in range(len(forward_gray_code)):
            for_sub_dict = forward_gray_code[i]
            for_num_control_qubits = len(for_sub_dict['control_value ='])
            for_new_target_info = []
            for_target_info = for_sub_dict['controls =']
            for_ctrl_value = for_sub_dict['control_value =']
            for j, k in enumerate(for_ctrl_value):
                if k == '0':
                    for_new_target = for_target_info[j]
                    for_new_target_info.append(for_new_target)
                    # change control value to be 1 if it is 0
                    forward_gray_code[i]['control_value ='] = [
                        '1']*for_num_control_qubits

            if len(for_new_target_info) != 0:
                # returns info about which two-level array gate decomposition
                #  needs pauli x to correct ctrl_value = 0
                for_array_gate_info_with_control_0.append(
                    [i, for_new_target_info])
                gate_keys.append(i)

        # create pauli arrays
        full_left_pauli_gate_dict = {}
        full_right_pauli_gate_dict = {}
        for i in range(len(for_array_gate_info_with_control_0)):
            left_pauli_array = []
            right_pauli_array = []
            gate_targets = for_array_gate_info_with_control_0[i][1]
            for j in range(len(gate_targets)):
                target_qubit = gate_targets[j]
                paulix = Gate("X", targets=target_qubit)
                left_pauli_array.append(paulix)
            key = gate_keys[i]
            full_left_pauli_gate_dict[key] = left_pauli_array
            # the pauli gate order is reversed when applied from the right
            right_pauli_array = left_pauli_array[::-1]
            full_right_pauli_gate_dict[key] = right_pauli_array

        # create full forward gate dict with pauli gates
        full_gate_dict = {}
        for i in range(len(forward_gray_code)):
            if len(for_array_gate_info_with_control_0) == len(
                        forward_gray_code):
                left_pauli = full_left_pauli_gate_dict[i]
                right_pauli = full_right_pauli_gate_dict[i]
                gray_code_with_ctrl_1 = forward_gray_code[i]
                full_gate_i_info = [
                    left_pauli, gray_code_with_ctrl_1, right_pauli]
                full_gate_dict[i] = full_gate_i_info
            elif len(for_array_gate_info_with_control_0) < len(
                    forward_gray_code):
                if i in gate_keys:
                    left_pauli = full_left_pauli_gate_dict[i]
                    right_pauli = full_right_pauli_gate_dict[i]
                    gray_code_with_ctrl_1 = forward_gray_code[i]
                    full_gate_i_info = [
                        left_pauli, gray_code_with_ctrl_1, right_pauli]
                    full_gate_dict[i] = full_gate_i_info
                else:
                    full_gate_dict[i] = forward_gray_code[i]

        # backward gray code info
        back_array_gate_info_with_control_0 = []
        # create gate keys list for backward gray code
        back_gate_keys = []
        for key in backward_gray_code.keys():
            back_gate_keys.append(key)

        # find qubits with ctrl_value = 0 and change them to 1
        for i in range(len(backward_gray_code)):
            key = back_gate_keys[i]
            back_sub_dict = backward_gray_code[key]
            back_num_control_qubits = len(back_sub_dict['control_value ='])
            back_new_target_info = []
            back_target_info = back_sub_dict['controls =']
            back_ctrl_value = back_sub_dict['control_value =']
            for j, k in enumerate(back_ctrl_value):
                if k == '0':
                    back_new_target = back_target_info[j]
                    back_new_target_info.append(back_new_target)
                    # change control value to be 1 if it is 0
                    backward_gray_code[key]['control_value ='] = [
                        '1']*back_num_control_qubits

            if len(back_new_target_info) != 0:
                # returns info about which two-level array gate decomposition
                # needs pauli x to correct ctrl_value = 0
                back_array_gate_info_with_control_0.append(
                    [key, back_new_target_info])

        # create pauli arrays
        back_full_left_pauli_gate_dict = {}
        back_full_right_pauli_gate_dict = {}
        for i in range(len(back_array_gate_info_with_control_0)):
            left_pauli_array = []
            right_pauli_array = []
            gate_targets = back_array_gate_info_with_control_0[i][1]
            for j in range(len(gate_targets)):
                target_qubit = gate_targets[j]
                paulix = Gate("X", targets=target_qubit)
                left_pauli_array.append(paulix)
            key = back_gate_keys[i]
            back_full_left_pauli_gate_dict[key] = left_pauli_array
            # the pauli gate order is reversed when applied from the right
            right_pauli_array = left_pauli_array[::-1]
            back_full_right_pauli_gate_dict[key] = right_pauli_array

        # create full forward gate dict with pauli gates
        back_full_gate_dict = {}
        for i in range(len(backward_gray_code)):
            if len(back_array_gate_info_with_control_0) == len(
                    backward_gray_code):
                key = back_gate_keys[i]
                left_pauli = back_full_left_pauli_gate_dict[key]
                right_pauli = back_full_right_pauli_gate_dict[key]
                gray_code_with_ctrl_1 = backward_gray_code[key]
                back_full_gate_i_info = [
                    left_pauli, gray_code_with_ctrl_1, right_pauli]
                back_full_gate_dict[key] = back_full_gate_i_info
            elif len(for_array_gate_info_with_control_0) < len(
                    forward_gray_code):
                key = back_gate_keys[i]
                if i in gate_keys:
                    left_pauli = back_full_left_pauli_gate_dict[key]
                    right_pauli = back_full_right_pauli_gate_dict[key]
                    gray_code_with_ctrl_1 = backward_gray_code[key]
                    back_full_gate_i_info = [
                        left_pauli, gray_code_with_ctrl_1, right_pauli]
                    back_full_gate_dict[key] = back_full_gate_i_info
                else:
                    back_full_gate_dict[key] = backward_gray_code[key]

        # calculated lengths
        len_forward_gate_info = len(full_gate_dict)
        len_backward_gate_info = len(back_full_gate_dict)

        if len_forward_gate_info == 0 and len_backward_gate_info == 0:
            return(full_gray_code_info)
        elif len_forward_gate_info == 0 and len_backward_gate_info > 0:
            return([forward_gray_code, back_full_gate_dict])
        elif len_forward_gate_info > 0 and len_backward_gate_info == 0:
            return([full_gate_dict, backward_gray_code])
        else:
            return([full_gate_dict, back_full_gate_dict])
