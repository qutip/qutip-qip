import numpy as np
import cmath
import pytest

from qutip import (
    Qobj, average_gate_fidelity, rand_unitary
)
from qutip_qip.circuit import QubitCircuit
from qutip_qip.decompose.decompose_two_qubit_gate import (
    decompose_two_qubit_to_two_level_unitary)


def test_two_level_output():
    two_qubit = rand_unitary(4, dims=[[2, 2], [2, 2]])
    dict_and_gate_list = decompose_two_qubit_to_two_level_unitary(two_qubit)
    quantum_circuit = QubitCircuit(2, reverse_states=False)
    user_gates_from_output = dict_and_gate_list[0]
    quantum_circuit.user_gates = user_gates_from_output
    quantum_circuit.add_gates(dict_and_gate_list[-1])
    calculatedu = quantum_circuit.compute_unitary()
    fid = average_gate_fidelity(calculatedu, two_qubit)
    assert np.isclose(fid, 1.0)
