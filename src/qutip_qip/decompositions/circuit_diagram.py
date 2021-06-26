from qutip_qip.decompositions.general_decompositions import (check_input,
check_input_shape, convert_qobj_gate_to_array)

from qutip_qip.decompositions.single_qubit_decompositions import (ZYZ_rotation)
from qutip import *
from qutip_qip.operations import *
from qutip_qip.circuit import QubitCircuit, Gate
from numpy import pi
import numpy as np

def decomposed_gate_to_circuit(decomposed_gate,num_of_qubits):
    if isinstance(decomposed_gate,tuple) == True:
        q_circuit = QubitCircuit(num_of_qubits, reverse_states=False)
        for i in decomposed_gate:
            q_circuit.add_gate(i)
        return(q_circuit)
    else:
        raise TypeError("The list of gates is not a tuple.")
