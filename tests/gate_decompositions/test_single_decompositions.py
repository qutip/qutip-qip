import numpy as np
import cmath
import pytest

from qutip.qobj import Qobj
from qutip.metrics import average_gate_fidelity


from qutip_qip.decompositions.single_decompositions import (_ZYZ_rotation, _ZXZ_rotation,
                                                            ABC_decomposition, decompose_to_rotation_matrices)

from qutip_qip.decompositions.decompositions_extras import (decomposed_gates_to_circuit, matrix_of_decomposed_gates)

# Fidelity closer to 1 means the two states are similar to each other
H = Qobj([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
sigmax = Qobj([[0,1],[1,0]])
sigmay = Qobj([[0,-1j],[1j,0]])
sigmaz = Qobj([[1,0],[0,-1]])
SQRTNOT = Qobj([[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]])
T = Qobj([[1,0],[0,cmath.rect(1, np.pi/4)]])
S = Qobj([[1,0],[0,1j]])

@pytest.mark.parametrize("gate",[H, sigmax, sigmay, sigmaz, SQRTNOT, S, T])
@pytest.mark.parametrize("method",[_ZYZ_rotation, _ZXZ_rotation, ABC_decomposition])
def test_single_qubit_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    target = 0
    gate_list = method(gate,1,target)
    decomposed_gates_circuit = decomposed_gates_to_circuit(gate_list,1)
    decomposed_gates_final_matrix = matrix_of_decomposed_gates(decomposed_gates_circuit)
    fidelity_of_input_output = average_gate_fidelity(gate, decomposed_gates_final_matrix)
    assert(np.isclose(fidelity_of_input_output,1.0))

@pytest.mark.parametrize("gate",[H, sigmax, sigmay, sigmaz, SQRTNOT, S, T])
@pytest.mark.parametrize("method",['ZXZ','ZYZ'])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    target = 0
    gate_list = decompose_to_rotation_matrices(gate,1,target,method)
    decomposed_gates_circuit = decomposed_gates_to_circuit(gate_list,1)
    decomposed_gates_final_matrix = matrix_of_decomposed_gates(decomposed_gates_circuit)
    fidelity_of_input_output = average_gate_fidelity(gate, decomposed_gates_final_matrix)
    assert(np.isclose(fidelity_of_input_output,1.0))

@pytest.mark.parametrize("gate",[H, sigmax, sigmay, sigmaz, SQRTNOT, S, T])
@pytest.mark.parametrize("method",[_ZYZ_rotation, _ZXZ_rotation, ABC_decomposition])
def test_output_is_tuple(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    target = 0
    gate_list = method(gate,1,target)
    assert(isinstance(gate_list, tuple))

@pytest.mark.parametrize("gate",[H, sigmax, sigmay, sigmaz, SQRTNOT, S, T])
@pytest.mark.parametrize("method",['ZXZ','ZYZ'])
def test_check_single_qubit_to_decompose_to_rotations(gate, method):
    """Initial matrix and product of final decompositions are same within some phase."""
    target = 0
    gate_list = decompose_to_rotation_matrices(gate,1,target,method)
    assert(isinstance(gate_list, tuple))
