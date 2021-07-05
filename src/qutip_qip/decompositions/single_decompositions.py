import numpy as np
import cmath

from qutip.qobj import Qobj
from qutip_qip.decompositions.general_decompositions import (check_input, check_input_shape,
convert_qobj_gate_to_array, extract_global_phase, MethodError, GateError)

from qutip_qip.operations import *
from qutip_qip.circuit import QubitCircuit, Gate


# Functions for decompose_to_rotation_matrices
single_qubit = 1
def _angles_for_ZYZ(input_gate, single_qubit):
    """ Finds and returns the angles for ZYZ rotation matrix. These are
    used to change ZYZ to other combinations.
    """
    global_phase_angle = extract_global_phase(input_gate,single_qubit)
    input_array = convert_qobj_gate_to_array(input_gate)
    input_array = input_array/cmath.rect(1,global_phase_angle)
    # separate all the elements
    a = input_array[0][0]
    b = input_array[0][1]
    a_star = input_array[1][1]
    b_star = input_array[1][0]

    # find alpha, beta and theta
    alpha = cmath.phase(-a_star) + cmath.phase(b_star)
    beta = cmath.phase(-a_star) - cmath.phase(b_star)
    theta = 2*np.arctan2(np.absolute(b_star), np.absolute(a))
    return(alpha, theta, beta, np.pi+global_phase_angle)



def _ZYZ_rotation(input_gate, num_of_qubits, target):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_y`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    angle_list = _angles_for_ZYZ(input_gate, single_qubit)
    alpha = angle_list[0]
    beta = angle_list[2]
    theta = angle_list[1]
    global_phase_angle=angle_list[3]

    # for string in circuit diagram
    alpha_string = alpha/np.pi
    beta_string = beta/np.pi
    theta_string = theta/np.pi
    global_phase_angle_string = global_phase_angle/np.pi

    Phase_gate = Gate("GLOBALPHASE",targets=[target], arg_value=global_phase_angle, arg_label=r'{:0.2f} \times \pi'.format(global_phase_angle_string))
    Rz_alpha = Gate("RZ",targets=[target], arg_value=alpha, arg_label=r'{:0.2f} \times \pi'.format(alpha_string))
    Ry_theta = Gate("RY",targets=[target], arg_value=theta, arg_label=r'{:0.2f} \times \pi'.format(theta_string))
    Rz_beta = Gate("RZ",targets=[target], arg_value=beta, arg_label=r'{:0.2f} \times \pi'.format(beta_string))

    return(Rz_alpha, Ry_theta, Rz_beta, Phase_gate)

def _ZXZ_rotation(input_gate, num_of_qubits, target):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_x`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    angle_list = _angles_for_ZYZ(input_gate, single_qubit)
    alpha = angle_list[0]
    alpha = alpha - np.pi/2
    beta = angle_list[2]
    beta = beta + np.pi/2
    theta = angle_list[1]
    global_phase_angle=angle_list[3]

    # for string in circuit diagram
    alpha_string = alpha/np.pi
    beta_string = beta/np.pi
    theta_string = theta/np.pi
    global_phase_angle_string = global_phase_angle/np.pi

    Phase_gate = Gate("GLOBALPHASE",targets=[target], arg_value=global_phase_angle, arg_label=r'{:0.2f} \times \pi'.format(global_phase_angle_string))
    Rz_alpha = Gate("RZ",targets=[target], arg_value=alpha, arg_label=r'{:0.2f} \times \pi'.format(alpha_string))
    Rx_theta = Gate("RX",targets=[target], arg_value=theta, arg_label=r'{:0.2f} \times \pi'.format(theta_string))
    Rz_beta = Gate("RZ",targets=[target], arg_value=beta, arg_label=r'{:0.2f} \times \pi'.format(beta_string))

    return(Rz_alpha, Rx_theta, Rz_beta, Phase_gate)


_rotation_matrices_dictionary ={"ZYZ": _ZYZ_rotation,
                                "ZXZ": _ZXZ_rotation,
                                }

def decompose_to_rotation_matrices(input_gate, num_of_qubits, target, method):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j`.

    Here, :math:`i \neq j` and :math:`i, j \in {x, y, z}`

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{R}_i(\alpha)  \textrm{R}_j(\theta)  \textrm{R}_i(\beta)

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

        .. list-table::
            :widths: auto
            :header-rows: 1

            * - Method Key
              - Method
            * - ZYZ
              - :math:`\textrm{R}_z(\alpha)  \textrm{R}_y(\theta)  \textrm{R}_z(\beta)`
            * - ZXZ
              - :math:`\textrm{R}_z(\alpha)  \textrm{R}_x(\theta)  \textrm{R}_z(\beta)`

    Returns
    -------
    tuple
        The gates in the decomposition are returned as a tuple of :class:`Gate`
        objects. This tuple will contain 4 elements per each :math:`1 \times 1`
        qubit gate - :math:`\textrm{R}_i(\alpha)`, :math:`\textrm{R}_j(\theta)`,
        :math:`\textrm{R}_i(\beta)`, and some global phase gate.
    """
    try:
        assert num_of_qubits == 1
    except AssertionError:
        if target is None and num_of_qubits >1:
            raise GateError("This method is valid for single qubit gates only. Provide a target qubit for single qubit gate.")


    key = _rotation_matrices_dictionary.keys()
    if str(method) in key:
        method = _rotation_matrices_dictionary[str(method)]
        return(method(input_gate, num_of_qubits, target))

    else:
        raise MethodError("Invalid method chosen.")


def ABC_decomposition(input_gate, num_of_qubits, target):
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
        if target is None and num_of_qubits >1:
            raise GateError("This method is valid for single qubit gates only. Provide a target qubit for larger circuits. ")


    global_phase_angle = extract_global_phase(input_gate,single_qubit)
    input_array = convert_qobj_gate_to_array(input_gate)
    input_array = input_array/cmath.rect(1,global_phase_angle)
    # separate all the elements
    a = input_array[0][0]
    b = input_array[0][1]
    a_star = input_array[1][1]
    b_star = input_array[1][0]

    global_phase_angle=np.pi+global_phase_angle
    global_phase_angle_string = global_phase_angle/np.pi
    # find alpha, beta and theta
    alpha = cmath.phase(-a_star) + cmath.phase(b_star)
    alpha_string = alpha/np.pi # for string in circuit diagram
    beta = cmath.phase(-a_star) - cmath.phase(b_star)
    beta_string = beta/np.pi
    theta = 2*np.arctan2(np.absolute(b_star), np.absolute(a))
    theta_string = theta/np.pi


    Phase_gate = Gate("GLOBALPHASE",targets=[0], arg_value=global_phase_angle, arg_label=r'{:0.2f} \times \pi'.format(global_phase_angle_string))
    Rz_A = Gate("RZ",targets=[target], arg_value=alpha, arg_label=r'{:0.2f} \times \pi'.format(alpha_string))
    Ry_A = Gate("RY",targets=[target], arg_value=theta/2, arg_label=r'{:0.2f} \times \pi'.format(theta_string/2))
    Pauli_X = Gate("X",targets=[target])
    Ry_B = Gate("RY",targets=[target], arg_value=-theta/2, arg_label=r'{:0.2f} \times \pi'.format(-theta_string/2))
    Rz_B = Gate("RZ",targets=[target], arg_value=-(alpha+beta)/2, arg_label=r'{:0.2f} \times \pi'.format(-(alpha_string+beta_string)/2))
    Rz_C = Gate("RZ",targets=[target], arg_value=(-alpha+beta)/2, arg_label=r'{:0.2f} \times \pi'.format((-alpha_string+beta_string)/2))

    return(Rz_A, Ry_A, Pauli_X, Ry_B, Rz_B, Pauli_X, Rz_C, Phase_gate)
