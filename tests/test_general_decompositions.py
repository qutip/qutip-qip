
import numpy as np
import cmath

from qutip.qobj import Qobj
from .general_decompositions import (check_input, check_input_shape,
convert_qobj_gate_to_array, extract_global_phase)

from qutip_qip.operations import *
from qutip_qip.circuit import QubitCircuit, Gate

@pytest.mark.parametrize("invalid_input",[np.array([[1,1],[1,1]]),([[1,1],[1,1]]),1.5,3,(1,2,3,4),np.array([[],[]]),([[],[]]),()])
def test_check_input_non_qobj(invalid_input):
    """Checks if correct value is returned or not when the input is not a Qobj.
    """
    with pytest.raises(TypeError,match="The input matrix is not a Qobj."):
        check_input(invalid_input)

@pytest.mark.parametrize("matrix",[Qobj([[1, 2], [3, 4],[5,6]])])
def test_non_square_matrix(matrix):
    """Rectangular Qobj are identified correctly."""
    with pytest.raises(ValueError, match="Input is not a square matrix."):
        check_input(matrix)

@pytest.mark.parametrize("matrix",[Qobj([[1,4,3]]),Qobj([[1]]),Qobj([[2]]),Qobj([[]])])
def test_1d_matrix(matrix):
    """A 1D object is identified as invalid quantum gate"""
    with pytest.raises(ValueError, match="A 1-D Qobj is not a valid quantum gate."):
        check_input(matrix)


# TO DO : CHECK FOR LARGER NUMBER OF QUBITS
@pytest.mark.parametrize("unitary",[Qobj([[1,0],[0,-1]])])
def test_check_input_non_qobj(unitary):
    """Checks if unitary innput is correctly identified.
    """
    assert(check_input(unitary)==True)

@pytest.mark.parametrize("non_unitary",[Qobj([[1,1],[0,1]])])
def test_check_input_non_qobj(non_unitary):
    """Checks if non-unitary input is correctly identified.
    """
    assert(check_input(non_unitary)==False)



@pytest.mark.parametrize("valid_input",[Qobj([[1,0,0],[0,1,0],[0,0,1]]),rx(np.pi/2,3),z_gate(3),t_gate(3)])
def test_one_qutrit_gates(valid_input):
    """Checks if Qobj is converted to a numpy array.
    """
    assert(isinstance(convert_qobj_gate_to_array(valid_input),np.ndarray))
