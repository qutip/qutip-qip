import numpy as np
import cmath


from qutip.qobj import Qobj

from qutip_qip.decompositions.general_decompositions import (check_input,
check_input_shape, convert_qobj_gate_to_array, extract_global_phase)

from qutip_qip.decompositions.decompositions_extras import (decomposed_gates_to_circuit, matrix_of_decomposed_gates)
from qutip_qip.operations import *

from qutip_qip.decompositions.single_decompositions import (_angles_for_ZYZ, ABC_decomposition)

def decompose_fully_controlled_on_first_qubit_to_ABC(input_gate, num_of_qubits, target_qubit, control_qubit):
    """ Decomposes a gate fully controlled on first qubit into CNOT and A, B, C
    gates from single-qubit ABC decomposition.

    # Taken from Lemma 5.1 of  https://arxiv.org/abs/quant-ph/9503016
    """
    try:
        assert num_of_qubits == 2
    except AssertionError:
        if target_qubit is None and num_of_qubits >2 and control_qubit is None:
            raise GateError("This method is valid for two qubit gates only. Provide a target/control qubit for larger circuits. ")

    # this condition will be updated to a two_level_matrix of particu;ar type
    # after a two_level function is added
    # the condition is incomplete
    if {input_gate[0][0]==1 and input_gate[1][1] == 1 and input_gate[2][2] !=0 and input_gate[3][3]!=0}:
        global_phase_angle = extract_global_phase(input_gate,num_of_qubits)
        input_array = convert_qobj_gate_to_array(input_gate)
        input_array = input_array/cmath.rect(1,global_phase_angle)
        input_gate_sub_matrix = np.array([[input_gate[2][2],input_gate[2][3]],[input_gate[3][2],input_gate[3][3]]])

        gates_from_ABC = ABC_decomposition(Qobj(input_gate_sub_matrix), 1, target)

        Phase_gate = gates_from_ABC[0]
        Rz_A = gates_from_ABC[1]
        Ry_A = gates_from_ABC[2]
        CNOT1 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])
        Ry_B = gates_from_ABC[4]
        Rz_B = gates_from_ABC[5]
        CNOT2 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])
        Rz_C = gates_from_ABC[-1]

        return(Rz_A, Ry_A, CNOT1, Ry_B, Rz_B,CNOT2, Rz_C, Phase_gate)


def decompose_fully_controlled_on_first_qubit_to_AB(input_gate, num_of_qubits, target_qubit, control_qubit):
    """ Decomposes a gate fully controlled on first qubit into CNOT and A, B
    gates from single-qubit ABC decomposition.

    # Taken from Lemma 5.4 of  https://arxiv.org/abs/quant-ph/9503016
    """
    try:
        assert num_of_qubits == 2
    except AssertionError:
        if target_qubit is None and num_of_qubits >2 and control_qubit is None:
            raise GateError("This method is valid for two qubit gates only. Provide a target/control qubit for larger circuits. ")

    # this condition will be updated to a two_level_matrix of particu;ar type
    # after a two_level function is added
    # the condition is incomplete
    if {input_gate[0][0]==1 and input_gate[1][1] == 1 and input_gate[2][2] !=0 and input_gate[3][3]!=0:
        # add condition for 1 x 1 gate being product of Rx, Ry, Rz

        global_phase_angle = extract_global_phase(input_gate,num_of_qubits)
        input_array = convert_qobj_gate_to_array(input_gate)
        input_array = input_array/cmath.rect(1,global_phase_angle)
        input_gate_sub_matrix = np.array([[input_gate[2][2],input_gate[2][3]],[input_gate[3][2],input_gate[3][3]]])

        gates_from_ABC = ABC_decomposition(Qobj(input_gate_sub_matrix), 1, target)

        Phase_gate = gates_from_ABC[0]
        Rz_A = gates_from_ABC[1]
        Ry_A = gates_from_ABC[2]
        CNOT1 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])
        Ry_B = gates_from_ABC[4]
        Rz_B = gates_from_ABC[5]
        CNOT2 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])

        return(Rz_A, Ry_A, CNOT1, Ry_B, Rz_B,CNOT2, Phase_gate)


def decompose_fully_controlled_on_first_qubit_to_AB_one_CNOT(input_gate, num_of_qubits, target_qubit, control_qubit):
    """ Decomposes a gate fully controlled on first qubit into one CNOT and A, B
    gates from single-qubit ABC decomposition.

    # Taken from Lemma 5.5 of  https://arxiv.org/abs/quant-ph/9503016
    """
    try:
        assert num_of_qubits == 2
    except AssertionError:
        if target_qubit is None and num_of_qubits >2 and control_qubit is None:
            raise GateError("This method is valid for two qubit gates only. Provide a target/control qubit for larger circuits. ")

    # this condition will be updated to a two_level_matrix of particu;ar type
    # after a two_level function is added
    # the condition is incomplete
    if {input_gate[0][0]==1 and input_gate[1][1] == 1 and input_gate[2][2] !=0 and input_gate[3][3]!=0:
        # add condition for 1 x 1 gate being product of Rz, Ry, Rz
        global_phase_angle = extract_global_phase(input_gate,num_of_qubits)
        input_array = convert_qobj_gate_to_array(input_gate)
        input_array = input_array/cmath.rect(1,global_phase_angle)
        input_gate_sub_matrix = np.array([[input_gate[2][2],input_gate[2][3]],[input_gate[3][2],input_gate[3][3]]])

        gates_from_ABC = ABC_decomposition(Qobj(input_gate_sub_matrix), 1, target)

        Phase_gate = gates_from_ABC[0]
        Rz_A = gates_from_ABC[1]
        Ry_A = gates_from_ABC[2]
        CNOT1 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])
        Ry_B = gates_from_ABC[4]
        Rz_B = gates_from_ABC[5]

def decompose_two_qubit_diagonal(input_gate, num_of_qubits, target_qubit, control_qubit):
    """ Decomposes a gate fully controlled on first qubit into CNOT and Rz

    # Taken from Corollary 5.3 of  https://arxiv.org/abs/quant-ph/9503016
    # Also in Proposition 2.1 of https://arxiv.org/abs/quant-ph/0211002
    """
    try:
        assert num_of_qubits == 2
    except AssertionError:
        if target_qubit is None and num_of_qubits >2 and control_qubit is None:
            raise GateError("This method is valid for two qubit gates only. Provide a target/control qubit for larger circuits. ")

    # this condition will be updated to a two_level_matrix of particu;ar type
    # after a two_level function is added
    # the condition is incomplete
    if {input_gate[0][0]!=0 and input_gate[1][1] != 0 and input_gate[2][2] !=0 and input_gate[3][3]!=0:
        # add condition for 1 x 1 gate being product of Rz, Ry, Rz
        global_phase_angle = extract_global_phase(input_gate,num_of_qubits)
        input_array = convert_qobj_gate_to_array(input_gate)
        input_array = input_array/cmath.rect(1,global_phase_angle)
        # upto this could be defined in two_level function
        input_gate_sub_matrix = np.array([[input_gate[2][2],input_gate[2][3]],[input_gate[3][2],input_gate[3][3]]])


        z1 = input_gate[0][0]
        z2 = input_gate[1][1]
        z3 = input_gate[2][2]
        z4 = input_gate[3][3]

        z_product = z1 * (z2**(-1)) * (z3**(-1)) * z4

        phi = -cmath.phase(z_product)
        alpha = _angles_for_ZYZ(Qobj(input_gate_sub_matrix), 1)[0]
        beta = _angles_for_ZYZ(input_gate_sub_matrix, 1)[2]
        global_phase_angle = _angles_for_ZYZ(input_gate_sub_matrix, 1)[-1]

        # for string in circuit diagram
        alpha_string = alpha/np.pi
        beta_string = beta/np.pi
        phi_string = phi/np.pi
        global_phase_angle_string = global_phase_angle/np.pi


        Phase_gate = Phase_gate = Gate("GLOBALPHASE",targets=[target], arg_value=global_phase_angle, arg_label=r'{:0.2f} \times \pi'.format(global_phase_angle_string))
        CNOT1 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])
        Rz_phi = Gate("RZ",targets=[target], arg_value=phi, arg_label=r'{:0.2f} \times \pi'.format(phi_string))
        CNOT2 = Gate("CNOT", controls=[control_qubit], targets=[target_qubit])
        Rz_alpha = Gate("RZ",targets=[target_qubit], arg_value=alpha, arg_label=r'{:0.2f} \times \pi'.format(alpha_string))
        Rz_beta = Gate("RZ",targets=[control_qubit], arg_value=beta, arg_label=r'{:0.2f} \times \pi'.format(beta_string))

        return(CNOT1, Rz_phi, CNOT2, Rz_alpha, Rz_beta)
