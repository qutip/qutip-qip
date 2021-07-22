import numpy as np
import cmath

from qutip import Qobj
from qutip_qip.decompose._utility import (
    check_gate,
    MethodError,
)

import warnings
from qutip_qip.circuit import Gate
from qutip_qip.operations import controlled_gate

from .decompose_single_qubit_gate import decompose_one_qubit_gate

# for unknown labels in two level gates
warnings.filterwarnings("ignore", category=UserWarning)


def _U6(single_qubit_U, control=0, target=1, control_value=1):
    """Defines a two-level unitary whose decomposition is known.
    Lemma 5.1 of https://arxiv.org/abs/quant-ph/9503016
    """
    check_gate(single_qubit_U, num_qubits=1)
    two_level_U6 = controlled_gate(
        single_qubit_U, N=2, control=0, target=1, control_value=1
    )
    return two_level_U6


def _U5(single_qubit_U, control=1, target=0, control_value=1):
    """Defines a two-level unitary whose decomposition is known.
    U6 but with the control and target qubits switched.
    """
    check_gate(single_qubit_U, num_qubits=1)
    two_level_U5 = controlled_gate(
        single_qubit_U, N=2, control=1, target=0, control_value=1
    )
    return two_level_U5


def _decompose_U6(input_array):
    """Decomposes a two-level unitary into ABC and CNOT when target =1 and
    the control value is 1.
    Based on Lemma 5.1 of arXiv:quant-ph/9503016v1
    """
    U = np.array(
        [[input_array[2][2], input_array[2][3]],
            [input_array[3][2], input_array[3][3]]]
    )
    check_gate(Qobj(U), num_qubits=1)
    gate_list = decompose_one_qubit_gate(Qobj(U), "ZYZ_PauliX")

    # Change the gate targets for ABC and X gates from default = 0
    # First 2 indices are for A, next 2 are for B and last is for C
    # Note that we have avoided changing the Pauli X indices because it is
    # easier to define a gate with the target qubit.
    target_indices_to_change = [0, 1, 3, 4, 6]
    for i in target_indices_to_change:
        gate_list[i].targets = [1]

    # last is phase whose target is not changed based on Lemma 5.2 of
    # arXiv:quant-ph/9503016v1 but the gate type is changed
    phase_angle = gate_list[-1].arg_value
    phase_gate = Gate(
        "PHASEGATE",
        targets=[0],
        arg_value=phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(phase_angle / np.pi),
    )
    Pauli_X = Gate("X", targets=[1], classical_controls=[0])

    CNOT_ctrl_0 = Gate("CNOT", controls=0, targets=1)
    return (
        gate_list[0],
        gate_list[1],
        Pauli_X,
        CNOT_ctrl_0,
        Pauli_X,
        gate_list[3],
        gate_list[4],
        Pauli_X,
        CNOT_ctrl_0,
        Pauli_X,
        gate_list[6],
        phase_gate,
    )


def _decompose_U5(input_array):
    """Decomposes a two-level unitary into ABC and CNOT when target = 0 and
    control value is 1.
    Based on Lemma 5.1 of arXiv:quant-ph/9503016v1 with flipped targets.
    """
    U = np.array(
        [[input_array[1][1], input_array[1][3]],
            [input_array[3][1], input_array[3][3]]]
    )
    check_gate(Qobj(U), num_qubits=1)
    gate_list = decompose_one_qubit_gate(Qobj(U), "ZYZ_PauliX")

    # No need to change targets of A, B, C - default is already 0
    # last is phase whose target is changed based on Lemma 5.2 of
    # arXiv:quant-ph/9503016v1 alongwith the gate type
    phase_angle = gate_list[-1].arg_value
    phase_gate = Gate(
        "PHASEGATE",
        targets=[1],
        arg_value=phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(phase_angle / np.pi),
    )
    Pauli_X = Gate("X", targets=[0], classical_controls=[0])

    CNOT_ctrl_1 = Gate("CNOT", controls=1, targets=0)
    return (
        gate_list[0],
        gate_list[1],
        Pauli_X,
        CNOT_ctrl_1,
        Pauli_X,
        gate_list[3],
        gate_list[4],
        Pauli_X,
        CNOT_ctrl_1,
        Pauli_X,
        gate_list[6],
        phase_gate,
    )


def _decompose_to_two_level_arrays(input_gate):
    """Output of two qubit gate in terms of two-level numpy arrays.
    """
    check_gate(input_gate, num_qubits=2)
    input_array = input_gate.full()

    # Calculate the two level numpy arrays
    array_list = []
    index_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
    for i in range(len(index_list)):
        index_1, index_2 = index_list[i]

        # Values of single qubit U forming the two level unitary
        a = input_array[index_1][index_1]
        a_star = np.conj(a)
        b = input_array[index_2][index_1]
        b_star = np.conj(b)
        norm_constant = cmath.sqrt(np.absolute(a*a_star)+np.absolute(b*b_star))

        # Create identity array and then replace with above values for index_1
        # and index_2
        U_two_level = np.identity(4, dtype=complex)
        U_two_level[index_1][index_1] = a_star/norm_constant
        U_two_level[index_2][index_1] = b/norm_constant
        U_two_level[index_1][index_2] = b_star/norm_constant
        U_two_level[index_2][index_2] = -a/norm_constant

        # Change input by multiplying by above two-level
        input_array = np.dot(U_two_level, input_array)

        # U dagger to calculate the gates
        U__two_level_dagger = np.transpose(np.conjugate(U_two_level))
        U__two_level_dagger = Qobj(U__two_level_dagger, dims=[[2, 2], [2, 2]])
        array_list.append(U__two_level_dagger)

    # for U6 - multiply input array by U5 and take dagger
    U6_dagger = input_array
    array_list.append(Qobj(U6_dagger, dims=[[2, 2], [2, 2]]))
    return(array_list)


def decompose_two_qubit_to_two_level_unitary(input_gate):
    """Decomposes input two-qubit unitary into two-level unitary gate objects.
    Returns
    ----------
    tuple: (two_level_dictionary, two_level_gates)
    two_level_dictionary
        Dictionary containing string labels to insert user-defined gates in
        a quantum circuit.
    two_level_gates
        Gates describing the decomposition of the two-qubit input gate.
    """
    check_gate(input_gate, num_qubits=2)
    array_list = _decompose_to_two_level_arrays(input_gate)
    user_gates = {
        "U1": array_list[0],
        "U2": array_list[1],
        "U3": array_list[2],
        "U4": array_list[3],
        "U5": array_list[4],
        "U6": array_list[5],
        }

    U1_gate = Gate("U1", targets=[0, 1])
    U2_gate = Gate("U2", targets=[0, 1])
    U3_gate = Gate("U3", targets=[1, 0])
    U4_gate = Gate("U4", targets=[0, 1])
    U5_gate = Gate("U5", targets=[0, 1])
    U6_gate = Gate("U6", targets=[0, 1])

    gate_list = (U6_gate, U5_gate, U4_gate, U3_gate, U2_gate, U1_gate)

    return(user_gates, gate_list)


def decompose_two_qubit_to_CNOT_and_single_qubit_gates(input_gate):
    """ Decomposes a two qubit gate into a decomposition described by CNOT,
    Pauli X and single qubit rotation matrices (ZYZ).
    """
    check_gate(input_gate, num_qubits=2)
    two_level_arrays = _decompose_to_two_level_arrays(input_gate)

    all_gates_in_circuit = []

    paulix0 = Gate("X", targets=[0], classical_controls=[0])
    paulix1 = Gate("X", targets=[1], classical_controls=[0])
    CNOT01 = Gate("CNOT", controls=0, targets=1)
    CNOT10 = Gate("CNOT", controls=1, targets=0)

    # U6 gates
    U6_gates_list = []
    U6_array = np.array(two_level_arrays[-1])
    U6_gates = _decompose_U6(U6_array)
    for i in range(len(U6_gates)):
        U6_gates_list.append(U6_gates[i])

    all_gates_in_circuit.extend(U6_gates_list)

    # U5 gates
    U5_gates_list = []
    U5_array = np.array(two_level_arrays[-2])
    U5_sub_array = np.array(
        [[U5_array[1][1], U5_array[1][3]], [U5_array[3][1], U5_array[3][3]]])
    U5_to_U6 = _U6(Qobj(U5_sub_array), control=0, target=1, control_value=1)
    U5_gates = _decompose_U6(U5_to_U6.full())
    for i in range(len(U5_gates)):
        U5_gates_list.append(U5_gates[i])

    all_gates_in_circuit.extend(U5_gates_list)

    # U4 gates
    U4_controls = [paulix1, paulix0, CNOT01, paulix0]
    U4_gates_list = [paulix0, CNOT01, paulix0, paulix1]
    U4_array = np.array(two_level_arrays[3])
    U4_sub_array = np.array(
        [[U4_array[1][1], U4_array[1][2]], [U4_array[2][1], U4_array[2][2]]])
    U4_to_U5 = _U5(Qobj(U4_sub_array), control=1, target=0, control_value=1)
    U4_sub_gate = _decompose_U5(U4_to_U5.full())
    for i in range(len(U4_sub_gate)):
        U4_gates_list.append(U4_sub_gate[i])

    U4_gates_list.extend(U4_controls)
    all_gates_in_circuit.extend(U4_gates_list)

    # U3 gates
    U3_controls = [paulix0, CNOT01, paulix0]
    U3_gates_list = [paulix0, CNOT01, paulix0]
    U3_array = np.array(two_level_arrays[2])
    U3_sub_array = np.array(
        [[U3_array[0][0], U3_array[0][3]], [U3_array[3][0], U3_array[3][3]]])
    U3_to_U6 = _U6(Qobj(U3_sub_array), control=0, target=1, control_value=1)
    U3_sub_gate = _decompose_U6(U3_to_U6.full())
    for i in range(len(U3_sub_gate)):
        U3_gates_list.append(U3_sub_gate[i])

    U3_gates_list.extend(U3_controls)
    all_gates_in_circuit.extend(U3_gates_list)
    # U2 gates
    U2_controls = [CNOT10, paulix0, CNOT01, paulix0]
    U2_gates_list = [paulix0, CNOT01, paulix0, CNOT10]
    U2_array = np.array(two_level_arrays[1])
    U2_sub_array = np.array(
        [[U2_array[0][0], U2_array[0][2]], [U2_array[2][0], U2_array[2][2]]])
    U2_to_U6 = _U6(Qobj(U2_sub_array), control=0, target=1, control_value=1)
    U2_sub_gate = _decompose_U6(U2_to_U6.full())
    for i in range(len(U2_sub_gate)):
        U2_gates_list.append(U2_sub_gate[i])

    U2_gates_list.extend(U2_controls)
    all_gates_in_circuit.extend(U2_gates_list)
    # U1 Gates
    U1_gates_list = [paulix0]
    U1_array = np.array(two_level_arrays[0])
    U1_sub_array = np.array(
        [[U1_array[0][0], U1_array[0][1]], [U1_array[1][0], U1_array[1][1]]])
    U1_to_U6 = _U6(Qobj(U1_sub_array), control=0, target=1, control_value=1)
    U1_sub_gate = _decompose_U6(U1_to_U6.full())
    for i in range(len(U1_sub_gate)):
        U1_gates_list.append(U1_sub_gate[i])

    U1_gates_list.append(paulix0)
    all_gates_in_circuit.extend(U1_gates_list)
    return(all_gates_in_circuit)
