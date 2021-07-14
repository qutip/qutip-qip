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


def check_gate(gate, num_qubits):
    """Verifies input is a valid quantum gate.

    Parameters
    ----------
    gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    num_qubits:
        Number of qubits in the circuit.
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


def extract_global_phase(input_gate, num_qubits):
    """ Extracts some common constant from all the elements of the matrix. The output
    is retuned in the complex exponential form.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    input_array = input_gate.full()
    determinant_of_input = np.linalg.det(input_array)
    global_phase_angle = cmath.phase(determinant_of_input)
    global_phase_angle = global_phase_angle / (2**num_qubits)
    return global_phase_angle
