import numpy as np
import cmath

from qutip import Qobj

class MethodError(Exception):
    """When invalid method is chosen, this error is raised.
    """
    pass

class GateError(Exception):
    """When chosen method cannot be applied to the input gate, this error
    is raised.
    """
    pass

def check_input(input_gate):
    """Verifies input is a valid quantum gate.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    bool
        If the input is a valid quantum gate matrix, returns "True".
        In the case of "False" being returned, this function will ensure no
        decomposition scheme can proceed.
    """
    # check if input is a qobj
    qobj_check = isinstance(input_gate,Qobj)
    if qobj_check == False:
        raise TypeError("The input matrix is not a Qobj.")
    else:
        # check if input is square and a unitary
        input_shape = input_gate.shape
        # check if input is 1 row and 1 column matrix
        if input_shape[0]==1:
            raise ValueError("A 1-D Qobj is not a valid quantum gate.")
        # check if the input is a rectangular matrix
        if input_shape[0] != input_shape[1]:
            raise ValueError("Input is not a square matrix.")
        else:
            unitary_check = Qobj.check_isunitary(input_gate)
            return unitary_check


def check_input_shape(input_gate, num_of_qubits):
    """Check if the shape of input gate is valid to act on input number of qubits.

    If :math:`n` is the number of qubits in the circuit then a valid quantum gate
    acting on these qubits must be of dimension :math:`2^n \\times 2^n`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    num_of_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    bool
        Returns "True" if the shape of input gate is valid to act on the number
        of qubits in the circuit.

    """
    input_check_bool = check_input(input_gate)
    if input_check_bool == True:
        input_shape = input_gate.shape
        return input_shape[0] == 2**num_of_qubits
    else:
        raise ValueError("Input is not unitary.")


def convert_qobj_gate_to_array(input_gate):
    """Converts a valid unitary quantum gate to a numpy array.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    input_gate : `np.array`
        The input is returned as a converted numpy array.
    """
    input_check_bool = check_input(input_gate)
    if input_check_bool == True:
        input_to_array = Qobj.full(input_gate)
        return(input_to_array)
    else:
        raise ValueError("Input is not unitary.")


def extract_global_phase(input_gate, num_of_qubits):
    """ Extracts some common constant from all the elements of the matrix. The output
    is retuned in the complex exponential form.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    if check_input_shape(input_gate, num_of_qubits) == True:
        input_array = convert_qobj_gate_to_array(input_gate)
        determinant_of_input = np.linalg.det(input_array)
        global_phase_angle = cmath.phase(determinant_of_input)
        global_phase_angle = global_phase_angle/(2**num_of_qubits)
        return(global_phase_angle)
    else:
        raise GateError("Gate shape does not match to the number of qubits in the circuit. ")
