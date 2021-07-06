import numpy as np
import cmath


from qutip.qobj import Qobj

from qutip_qip.decompositions.general_decompositions import (check_input,
check_input_shape, convert_qobj_gate_to_array, extract_global_phase)

from qutip_qip.decompositions.decompositions_extras import (decomposed_gates_to_circuit, matrix_of_decomposed_gates)
from qutip_qip.operations import *

def two_level_unitary_matrix(total_number_of_qubits, list_of_sub_matrix_values):
    """ Returns a numpy array obtained from a d-dimensional identity matrix and
    a 2 x 2 submatrix with non-trivial values.
    """
