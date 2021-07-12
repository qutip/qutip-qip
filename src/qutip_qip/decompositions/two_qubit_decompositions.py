import numpy as np
import cmath


from qutip.qobj import Qobj

from qutip_qip.decompositions.general_decompositions import (check_input,
check_input_shape, convert_qobj_gate_to_array, extract_global_phase)

from qutip_qip.decompositions.decompositions_extras import (decomposed_gates_to_circuit, matrix_of_decomposed_gates)
from qutip_qip.operations import *

from qutip_qip.decompositions.single_decompositions import (_angles_for_ZYZ, ABC_decomposition)

def _decompose_to_CNOT_ABC(input_gate, num_of_qubits, target_qubit, control_qubit):
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


def _decompose_to_two_CNOT_AB(input_gate, num_of_qubits, target_qubit, control_qubit):
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


def _decompose_to_one_CNOT_AB(input_gate, num_of_qubits, target_qubit, control_qubit):
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

_two_qubit_methods_dictionary ={"CNOT_ABC": _decompose_to_CNOT_ABC,
                                "two_CNOT_AB": _decompose_to_two_CNOT_AB,
                                "one_CNOT_AB": _decompose_to_one_CNOT_AB,
                                }

def decompose_two_qubit_gate(input_gate, num_of_qubits, target, control, method):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j` and CNOT gates.

    Here, :math:`i \neq j` and :math:`i, j \in {x, y, z}`

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    num_of_qubits : int
        Number of qubits being acted upon by input gate

    target : int
        If the circuit contains more than 1 qubits then provide target for
        single qubit gate.

    method : string
        Name of the preferred decompositions method

    Returns
    -------
    tuple
        The gates in the decomposition are returned as a tuple of :class:`Gate`
        objects. This tuple will contain 4 elements per each :math:`1 \times 1`
        qubit gate - :math:`\textrm{R}_i(\alpha)`, :math:`\textrm{R}_j(\theta)`,
        :math:`\textrm{R}_i(\beta)`, and some global phase gate.
    """
    try:
        assert num_of_qubits == 2
    except AssertionError:
        if target is None and num_of_qubits >2:
            raise GateError("This method is valid for single qubit gates only. Provide a target qubit for single qubit gate.")


    key = _two_qubit_methods_dictionary.keys()
    if str(method) in key:
        method = _two_qubit_methods_dictionary[str(method)]
        input_array = convert_qobj_gate_to_array(input_gate)

        # To make zeroes at all elements of first column and row besides 00

        # make [1,0]==0
        S1 = input_array[0, 0]**2 + input_array[1, 0]**2
        N1 = np.absolute(np.sqrt(S1))
        U1 = np.array([[np.conj(input_array[0, 0])/N1,np.conj(input_array[1, 0])/N1,0,0],[input_array[1, 0]/N1,-input_array[0, 0]/N1,0,0],[0,0,1,0],[0,0,0,1]])
        U1_U = np.matmul(U1,U)
        U1_dagger = np.matrix.H(U1)
        # This will change based on the type of control CNOT gate has
        # Might have to add Pauli X
        U1_gates = method(Qobj(U1_dagger), num_of_qubits, target, control)

        all_gates = []
        all_gates.append(U1_gates)

        # making [2,0] ==0
        S2 = U1_U[0, 0]**2 + U1_U[2, 0]**2
        N2 = np.absolute(np.sqrt(S2))
        U2 = np.array([[np.conj(U1_U[0, 0])/N2,0,np.conj(U1_U[2, 0])/N2,0],[0,1,0,0],[U1_U[2, 0]/N2,0,-U1_U[0, 0]/N2,0],[0,0,0,1]])
        U2U1_U = np.matmul(U2,U2U1_U)
        U2_dagger = np.matrix.H(U2)
        # Missing CNOT0 + Pauli X if needed
        all_gates.append(Gate("CNOT", controls=[control], targets=[target]))
        U2_gates = method(Qobj(U2_dagger), num_of_qubits, target, control)
        all_gates.append(U2_gates)
        all_gates.append(Gate("CNOT", controls=[control], targets=[target]))

        # making [3,0] ==0
        S3 = U2U1_U[0, 0]**2 + U2U1_U[3, 0]**2
        N3 = np.absolute(np.sqrt(S3))
        U3 = np.array([[np.conj(U2U1_U[0, 0])/N3,0,0,np.conj(U2U1_U[3, 0])/N3],[0,1,0,0],[0,0,1,0],[U2U1_U[3, 0]/N3,0,0,-U2U1_U[0, 0]/N3]])
        U3U2U1_U = np.matmul(U3,U2U1_U)
        U3_dagger = np.matrix.H(U3)
        # Missing Pauli X or use CNOT0
        all_gates.append(Gate("CNOT", controls=[target], targets=[control]))
        U3_gates = method(Qobj(U3_dagger), num_of_qubits, control, target) # target and control are reversed
        all_gates.append(U3_gates)
        all_gates.append(Gate("CNOT", controls=[target], targets=[control]))


        # making [1,2] ==0
        S4 = U3U2U1_U[0, 0]**2 + U3U2U1_U[1, 2]**2
        N4 = np.absolute(np.sqrt(S4))
        U4 = np.array([[1,0,0,0],[0,np.conj(U3U2U1_U[1, 1])/N4,np.conj(U3U2U1_U[2, 1])/N4,0],[0,U3U2U1_U[2, 1]/N4,-U3U2U1_U[1, 1]/N4,0],[0,0,0,1]])
        U4U3U2U1_U = np.matmul(U4,U3U2U1_U)
        U4_dagger = np.matrix.H(U4)
        # Missing Pauli X or use CNOT0
        all_gates.append(Gate("CNOT", controls=[target], targets=[control]))
        # Missing Pauli for U4_dagger
        U4_gates = method(Qobj(U4_dagger), num_of_qubits, control,target)
        all_gates.append(U4_gates)
        all_gates.append(Gate("CNOT", controls=[target], targets=[control]))

        # making [1,3] ==0
        S5 = U3U2U1_U[0, 0]**2 + U3U2U1_U[1, 3]**2
        N5 = np.absolute(np.sqrt(S4))
        U5 = np.array([[1,0,0,0],[0,np.conj(U3U2U1_U[1, 1])/N5,0, np.conj(U3U2U1_U[3, 1])/N5,0],[0,0,1,0],[0,U3U2U1_U[3, 1]/N5,0,-U3U2U1_U[1, 1]/N5]])
        U5U4U3U2U1_U = np.matmul(U5,U4U3U2U1_U)
        U5_dagger = np.matrix.H(U5)
        U5_gates = method(Qobj(U5_dagger), num_of_qubits,target, control)
        all_gates.append(U5_gates)

        # final two-level dagger
        U6_dagger = np.matrix.H(U5U4U3U2U1_U)
        U6_gates = method(Qobj(U6_dagger), num_of_qubits, control, target) # target and ocntrol qubits are flipped
        all_gates.append(U6_gates)

        return(all_gates)

    else:
        raise MethodError("Invalid method chosen.")


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
