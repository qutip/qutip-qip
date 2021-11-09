from qutip import Qobj


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
