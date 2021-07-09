
import numpy as np
import cmath
import pytest

from qutip.qobj import Qobj
from qutip_qip.decompose._utility import (check_input, check_input_shape,
convert_qobj_gate_to_array, extract_global_phase, MethodError, GateError)
from qutip_qip.operations import rx, z_gate, t_gate
from qutip_qip.circuit import QubitCircuit, Gate

# Tests for check_input_shape

@pytest.mark.parametrize("invalid_input",[np.array([[1,1],[1,1]]),([[1,1],[1,1]]),1.5,3,(1,2,3,4),np.array([[],[]]),([[],[]]),()])
def test_check_input_non_qobj(invalid_input):
    """Checks if correct value is returned or not when the input is not a Qobj.
    """
    with pytest.raises(TypeError,match="The input matrix is not a Qobj."):
        check_input(invalid_input)

@pytest.mark.parametrize("matrix",[Qobj([[1, 2], [3, 4],[5,6]])])
def test_check_input_non_square_matrix(matrix):
    """Rectangular Qobj are identified correctly."""
    with pytest.raises(ValueError, match="Input is not a square matrix."):
        check_input(matrix)

@pytest.mark.parametrize("matrix",[Qobj([[1,4,3]]),Qobj([[1]]),Qobj([[2]]),Qobj([[]])])
def test_check_input_1d_matrix(matrix):
    """A 1D object is identified as invalid quantum gate"""
    with pytest.raises(ValueError, match="A 1-D Qobj is not a valid quantum gate."):
        check_input(matrix)


# TO DO : CHECK FOR LARGER NUMBER OF QUBITS
@pytest.mark.parametrize("unitary",[Qobj([[1,0],[0,-1]])])
def test_check_input_valid_qobj(unitary):
    """Checks if unitary innput is correctly identified.
    """
    assert(check_input(unitary)==True)

@pytest.mark.parametrize("non_unitary",[Qobj([[1,1],[0,1]])])
def test_check_input_non_qobj(non_unitary):
    """Checks if non-unitary input is correctly identified.
    """
    assert(check_input(non_unitary)==False)

# Tests for check_input_shape
@pytest.mark.parametrize("unitary",[Qobj([[1,0],[0,-1]])])
def test_check_input_shape_unitary_input(unitary):
    """Checks if shape of input is correctly identified.
    """
    assert(check_input_shape(unitary,1)==True)

@pytest.mark.parametrize("non_unitary",[Qobj([[1,1],[0,1]])])
def test_check_input_non_qobj(non_unitary):
    """Checks if non-unitary input is correctly identified.
    """
    with pytest.raises(ValueError, match="Input is not unitary."):
        check_input_shape(non_unitary,1)

# Tests for convert_qobj_gate_to_array
@pytest.mark.parametrize("valid_input",[Qobj([[1,0,0],[0,1,0],[0,0,1]]),rx(np.pi/2,3),z_gate(3),t_gate(3)])
def test_one_qutrit_gates(valid_input):
    """Checks if Qobj is converted to a numpy array.
    """
    assert(isinstance(convert_qobj_gate_to_array(valid_input),np.ndarray))

@pytest.mark.parametrize("non_unitary",[Qobj([[1,1],[0,1]])])
def test_convert_qobj_gate_to_array(non_unitary):
    """Checks if non-unitary input is correctly identified.
    """
    with pytest.raises(ValueError, match="Input is not unitary."):
        convert_qobj_gate_to_array(non_unitary)

# Tests for extract_global_phase
def test_extract_global_phase_valid_input():
    """Checks if global phase is correctly identified for multiplication.
    """
    H = Qobj([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
    H_global_phase = extract_global_phase(H,1)
    assert(H_global_phase == np.pi/2)

def test_extract_global_phase_valid_input_incorrect_number_of_qubits():
    """Checks if global phase is correctly identified for multiplication.
    """
    H = Qobj([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
    with pytest.raises(GateError, match="Gate shape does not match to the number of qubits in the circuit. "):
        extract_global_phase(H,2)
