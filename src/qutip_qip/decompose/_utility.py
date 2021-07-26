from qutip import Qobj
import matplotlib.pyplot as plot
import numpy as np

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
    can be changed at each step.
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
    num_steps_gray_code = state_2_gray_code_index - state_1_gray_code_index

    # create a smaller gray code sequence between the two states of interest
    gray_code = _gray_code_sequence(num_qubits)[
        state_1_gray_code_index:state_2_gray_code_index+1]
    return(gray_code, num_steps_gray_code)


def gray_code_plot(index_of_state_1, index_of_state_2, num_qubits):
    """ Plots the difference between each step of a gray code sequence.

    Parameters
    -----------
    index_of_state_i: int
        The non-trivial indices in a two-level unitary matrix gate.

    num_qubits:
        Number of qubits being acted upon by the quantum gate.
    """
    # This function is still incomplete.


def gray_code_gate_info(index_of_state_1, index_of_state_2, num_qubits):
    """ Finds information about control and targets for CNOT gate in a gray
    code sequence.
    """
    gray_code_info = _gray_code_steps(
        index_of_state_1, index_of_state_2, num_qubits)
    gray_code = gray_code_info[0]
    num_steps = gray_code_info[1]

    # find number of ones in each gray code step, this will be used
    # to keep track of control values.
    one_bit_count = []
    for i in range(num_steps+1):
        check_bit = gray_code[i].count("1")
        one_bit_count.append(check_bit)

    gate_dictionary = {}
    for i in range(num_steps):
        a = gray_code[i]
        b = gray_code[i+1]
        for j in range(num_qubits):
            gate_controls = []
            gate_target = []
            gate_control_value = []
            while a[j] == b[j]:
                partial_gate_control = j
                gate_controls.append(partial_gate_control)
                gate_control_value.append(a[j])
            else:
                partial_gate_target = j
                gate_target.append(partial_gate_target)

        gate_dictionary[i] = [gate_controls, gate_control_value, gate_target]

    return(gate_dictionary)
