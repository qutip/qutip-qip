import numpy as np
import cmath

from qutip.qobj import Qobj
from .general_decompositions import (check_input, check_input_shape,
convert_qobj_gate_to_array, extract_global_phase)

from qutip_qip.operations import *
from qutip_qip.circuit import QubitCircuit, Gate

class MethodError():
    pass

class GateError():
    pass

def ZYZ_rotation(input_gate, num_of_qubits):
    """ An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`R_z` and :math:`R_y`.
    """
    try:
        assert num_of_qubits == 1
    except AssertionError:
        raise MethodError("This method is only valid for 1 qubit gates.")
    if check_input_shape(input_gate,num_of_qubits)==True:
        global_phase_angle = extract_global_phase(input_gate,1)
        global_phase_angle_string = global_phase_angle/np.pi

        input_array = np.dot(phasegate(global_phase_angle),convert_qobj_gate_to_array(input_gate))

        # separate all the elements
        a = input_array[0][0]
        b = input_array[0][1]
        a_star = input_array[1][1]
        b_star = input_array[1][0]

        # find alpha, beta and theta
        alpha = cmath.phase(a_star/a) + cmath.phase(-b_star/b)
        alpha_string = alpha/np.pi
        theta = 2*cmath.phase(complex(np.absolute(b),np.absolute(a)))
        theta_string = theta/np.pi
        beta = cmath.phase(a_star/a) - cmath.phase(-b_star/b)
        beta_string = beta/np.pi

        # input array was changed with a global gate created from
        # positive value of global phase
        # now undo it with a negative value of same phase angle global gate.
        Phase_gate = Gate("GLOBALPHASE",targets=[0], arg_value=-global_phase_angle, arg_label=r'{} \times \pi'.format(-global_phase_angle_string))
        Rz_alpha = Gate("RZ",targets=[0], arg_value=alpha, arg_label=r'{} \times \pi'.format(alpha_string))
        Ry_theta = Gate("RY",targets=[0], arg_value=theta, arg_label=r'{} \times \pi'.format(theta_string))
        Rz_beta = Gate("RZ",targets=[0], arg_value=beta, arg_label=r'{} \times \pi'.format(beta_string))

        return(Rz_alpha, Ry_theta, Rz_beta, Phase_gate)

    else:
        raise GateError("Shape of input does not match with the number of qubits.")

def ABC_decomposition(input_gate, num_of_qubits):
    """ An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`R_z` and :math:`R_y` and Pauli X.
    """
    try:
        assert num_of_qubits == 1
    except AssertionError:
        raise MethodError("This method is only valid for 1 qubit gates.")
    if check_input_shape(input_gate,num_of_qubits)==True:
        global_phase_angle = extract_global_phase(input_gate,1)
        global_phase_angle_string = global_phase_angle/np.pi

        input_array = np.dot(phasegate(global_phase_angle),convert_qobj_gate_to_array(input_gate))

        # separate all the elements
        a = input_array[0][0]
        b = input_array[0][1]
        a_star = input_array[1][1]
        b_star = input_array[1][0]

        # find alpha, beta and theta
        alpha = cmath.phase(a_star/a) + cmath.phase(-b_star/b)
        alpha_string = alpha/np.pi
        theta = 2*cmath.phase(complex(np.absolute(b),np.absolute(a)))
        theta_string = theta/np.pi
        beta = cmath.phase(a_star/a) - cmath.phase(-b_star/b)
        beta_string = beta/np.pi

        # input array was changed with a global gate created from
        # positive value of global phase
        # now undo it with a negative value of same phase angle global gate.
        Phase_gate = Gate("GLOBALPHASE",targets=[0], arg_value=-global_phase_angle, arg_label=r'{} \times \pi'.format(-global_phase_angle_string))
        Rz_A = Gate("RZ",targets=[0], arg_value=alpha, arg_label=r'{} \times \pi'.format(alpha_string))
        Ry_A = Gate("RY",targets=[0], arg_value=theta/2, arg_label=r'{} \times \pi'.format(theta_string/2))
        Pauli_X = Gate("X",targets=[0])
        Ry_B = Gate("RY",targets=[0], arg_value=-theta/2, arg_label=r'{} \times \pi'.format(-theta_string/2))
        Rz_B = Gate("RZ",targets=[0], arg_value=-(alpha+beta)/2, arg_label=r'{} \times \pi'.format(-(alpha_string+beta_string)/2))
        Rz_C = Gate("RZ",targets=[0], arg_value=(-alpha+beta)/2, arg_label=r'{} \times \pi'.format((-alpha_string+beta_string)/2))

        return(Rz_A, Ry_A, Pauli_X, Ry_B, Rz_B, Pauli_X, Rz_C, Phase_gate)

    else:
        raise GateError("Shape of input does not match with the number of qubits.")
