import numpy as np
import cmath
import pytest

from qutip.qobj import Qobj
from qutip_qip.operations import *
from qutip_qip.circuit import QubitCircuit, Gate

from qutip_qip.decompositions.general_decompositions import (check_input,
check_input_shape, convert_qobj_gate_to_array, extract_global_phase)

from qutip_qip.decompositions.single_decompositions import (_ZYZ_rotation, _ZXZ_rotation, _rotation_matrices_dictionary,
                                                            ABC_decomposition, decompose_to_rotation_matrices)

from qutip_qip.decompositions.decompositions_extras import (decomposed_gates_to_circuit, matrix_of_decomposed_gates)


# Tests for decomposed_gates_to_circuit
@pytest.mark.parametrize("invalid_input",[np.array([[1,1],[1,1]]),([[1,1],[1,1]]),1.5,3])
def test_decomposed_gates_to_circuit(invalid_input):
    """Checks if correct error is raised when input is anything but a tuple of gates.
    """
    with pytest.raises(TypeError,match="Input is not a tuple of gates."):
        decomposed_gates_to_circuit(invalid_input,1)

H = Qobj([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
sigmax = Qobj([[0,1],[1,0]])
H_zyz_gates = _ZYZ_rotation(H, 1, 0)
sigmax_zyz_gates = _ZYZ_rotation(sigmax, 1, 0)

@pytest.mark.parametrize("valid_input",[H_zyz_gates,sigmax_zyz_gates])
def test_decomposed_gates_to_circuit(valid_input):
    """Checks if output is of type QubitCircuit.
    """
    assert(isinstance(decomposed_gates_to_circuit(valid_input,1),QubitCircuit))


H_zyz_quantum_circuit = decomposed_gates_to_circuit(H_zyz_gates, 1)
sigmax_zyz_quantum_circuit = decomposed_gates_to_circuit(sigmax_zyz_gates, 1)
sigmax_zyz_output = (sigmax_zyz_quantum_circuit)
# Tests for matrix_of_decomposed_gates
@pytest.mark.parametrize("invalid_input",[np.array([[1,1],[1,1]]),([[1,1],[1,1]]),1.5,3])
def test_matrix_of_decomposed_gates(invalid_input):
    """Checks if correct error is raised when input is anything but a quantum circuit.
    """
    with pytest.raises(TypeError,match="Input is not of type QubitCircuit."):
        matrix_of_decomposed_gates(invalid_input)


@pytest.mark.parametrize("valid_input",[H_zyz_quantum_circuit,sigmax_zyz_quantum_circuit])
def test_matrix_of_decomposed_gates(valid_input):
    """Checks if final output is a Qobj.
    """
    final_output=matrix_of_decomposed_gates(valid_input)
    assert(isinstance(final_output, Qobj))
