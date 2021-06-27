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
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_y`.

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{R}_z(\alpha)  \textrm{R}_y(\theta)  \textrm{R}_z(\beta)

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    tuple
        The gates in the decomposition are returned as a tuple of :class:`Gate`
        objects. This tuple will contain 4 elements per each :math:`1 \times 1`
        qubit gate - :math:`\textrm{R}_z(\alpha)`, :math:`\textrm{R}_y(\theta)`,
        :math:`\textrm{R}_z(\beta)`, and some global phase gate.
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
        alpha = cmath.phase(-a_star) + cmath.phase(b_star)
        beta = cmath.phase(-a_star) - cmath.phase(b_star)
        alpha_string = alpha/np.pi
        beta_string = beta/np.pi

        theta = 2*np.arctan2(np.absolute(b_star), np.absolute(a))
        theta_string = theta/np.pi

        # input array was changed with a global gate created from
        # positive value of global phase
        Phase_gate = Gate("GLOBALPHASE",targets=[0], arg_value=global_phase_angle, arg_label=r'{:0.2f} \times \pi'.format(global_phase_angle_string))
        Rz_alpha = Gate("RZ",targets=[0], arg_value=alpha, arg_label=r'{:0.2f} \times \pi'.format(alpha_string))
        Ry_theta = Gate("RY",targets=[0], arg_value=theta, arg_label=r'{:0.2f} \times \pi'.format(theta_string))
        Rz_beta = Gate("RZ",targets=[0], arg_value=beta, arg_label=r'{:0.2f} \times \pi'.format(beta_string))

        return(Rz_alpha, Ry_theta, Rz_beta, Phase_gate)

    else:
        raise GateError("Shape of input does not match with the number of qubits.")

def ABC_decomposition(input_gate, num_of_qubits):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z`, :math:`\textrm{R}_y` and Pauli :math:`\textrm{X}`.

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{A} \textrm{X} \textrm{B} \textrm{X} \textrm{C}

    Where,
        - :math:`\textrm{A} = \textrm{R}_z(\alpha) \textrm{R}_y \left(\frac{\theta}{2} \right)`
        - :math:`\textrm{B} = \textrm{R}_y \left(\frac{-\theta}{2} \right) \textrm{R}_z \left(\frac{- \left(\alpha + \beta \right)}{2} \right)`
        - :math:`\textrm{C} = \textrm{R}_z \left(\frac{\left(-\alpha + \beta \right)}{2} \right)`


    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    tuple
        The gates in the decomposition are returned as a tuple of :class:`Gate`
        objects. This tuple will contain 6 elements per each :math:`1 \times 1`
        qubit gate - 2 gates forming :math:`\textrm{A}`, 2 gates forming :math:`\textrm{B}`,
        1 gates forming :math:`\textrm{C}`, and some global phase gate.
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
