import numpy as np
import cmath

from qutip import Qobj
from qutip_qip._decomposition_functions._utility import (
    check_gate,
    MethodError,
    GateError,
)

from qutip_qip.circuit import QubitCircuit, Gate


# Functions for decompose_to_rotation_matrices
def _angles_for_ZYZ(input_gate, num_qubits=1):
    """Finds and returns the angles for ZYZ rotation matrix. These are
    used to change ZYZ to other combinations.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The gate matrix that's supposed to be decomposed should be a Qobj.
    """
    check_gate(input_gate, num_qubits)
    input_array = input_gate.full()
    normalization_constant = np.sqrt(np.linalg.det(input_array))
    global_phase_angle = -cmath.phase(1 / normalization_constant)
    input_array = input_array * (1 / normalization_constant)

    # U = np.array([[a,b],[-b*,a*]])
    # If a = x+iy and b = p+iq, alpha = inv_tan(-y/x) - inv_tan(-q/p)
    a_negative = np.real(input_array[0][0]) - 1j * np.imag(input_array[0][0])
    b_negative = np.real(input_array[0][1]) - 1j * np.imag(input_array[0][1])

    # find alpha, beta and theta
    alpha = cmath.phase(a_negative) - cmath.phase(b_negative)
    beta = cmath.phase(a_negative) + cmath.phase(b_negative)
    theta = 2 * np.arctan2(np.absolute(b_negative), np.absolute(a_negative))

    return (alpha, -theta, beta, global_phase_angle)


def _ZYZ_rotation(input_gate, target, num_qubits=1):
    r"""An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_y`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    angle_list = _angles_for_ZYZ(input_gate, num_qubits)
    alpha = angle_list[0]
    beta = angle_list[2]
    theta = angle_list[1]
    global_phase_angle = angle_list[3]

    # for string in circuit diagram
    alpha_string = alpha / np.pi
    beta_string = beta / np.pi
    theta_string = theta / np.pi
    global_phase_angle_string = global_phase_angle / np.pi

    Phase_gate = Gate(
        "GLOBALPHASE",
        targets=[target],
        arg_value=global_phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(global_phase_angle_string),
    )
    Rz_beta = Gate(
        "RZ",
        targets=[target],
        arg_value=beta,
        arg_label=r"{:0.2f} \times \pi".format(beta_string),
    )
    Ry_theta = Gate(
        "RY",
        targets=[target],
        arg_value=theta,
        arg_label=r"{:0.2f} \times \pi".format(theta_string),
    )
    Rz_alpha = Gate(
        "RZ",
        targets=[target],
        arg_value=alpha,
        arg_label=r"{:0.2f} \times \pi".format(alpha_string),
    )

    return (Rz_alpha, Ry_theta, Rz_beta, Phase_gate)


def _ZXZ_rotation(input_gate, target, num_qubits=1):
    r"""An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_x`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    angle_list = _angles_for_ZYZ(input_gate, num_qubits)
    alpha = angle_list[0]
    alpha = alpha - np.pi / 2
    beta = angle_list[2]
    beta = beta + np.pi / 2
    theta = angle_list[1]
    global_phase_angle = angle_list[3]

    # for string in circuit diagram
    alpha_string = alpha / np.pi
    beta_string = beta / np.pi
    theta_string = theta / np.pi
    global_phase_angle_string = global_phase_angle / np.pi

    Phase_gate = Gate(
        "GLOBALPHASE",
        targets=[target],
        arg_value=global_phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(global_phase_angle_string),
    )
    Rz_alpha = Gate(
        "RZ",
        targets=[target],
        arg_value=alpha,
        arg_label=r"{:0.2f} \times \pi".format(alpha_string),
    )
    Rx_theta = Gate(
        "RX",
        targets=[target],
        arg_value=theta,
        arg_label=r"{:0.2f} \times \pi".format(theta_string),
    )
    Rz_beta = Gate(
        "RZ",
        targets=[target],
        arg_value=beta,
        arg_label=r"{:0.2f} \times \pi".format(beta_string),
    )

    return (Rz_alpha, Rx_theta, Rz_beta, Phase_gate)


# Functions for ABC_decomposition


def _ZYZ_pauli_X(input_gate, target, num_qubits=1):
    """Returns a 1 qubit unitary as a product of ZYZ rotation matrices and Pauli X."""
    angle_list = _angles_for_ZYZ(input_gate, num_qubits)
    alpha = angle_list[0]
    beta = angle_list[2]
    theta = angle_list[1]
    global_phase_angle = angle_list[3]

    # for string in circuit diagram
    alpha_string = alpha / np.pi
    beta_string = beta / np.pi
    theta_string = theta / np.pi
    global_phase_angle_string = global_phase_angle / np.pi

    Phase_gate = Gate(
        "GLOBALPHASE",
        targets=[0],
        arg_value=global_phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(global_phase_angle_string),
    )
    Rz_A = Gate(
        "RZ",
        targets=[target],
        arg_value=alpha,
        arg_label=r"{:0.2f} \times \pi".format(alpha_string),
    )
    Ry_A = Gate(
        "RY",
        targets=[target],
        arg_value=theta / 2,
        arg_label=r"{:0.2f} \times \pi".format(theta_string / 2),
    )
    Pauli_X = Gate("X", targets=[target])
    Ry_B = Gate(
        "RY",
        targets=[target],
        arg_value=-theta / 2,
        arg_label=r"{:0.2f} \times \pi".format(-theta_string / 2),
    )
    Rz_B = Gate(
        "RZ",
        targets=[target],
        arg_value=-(alpha + beta) / 2,
        arg_label=r"{:0.2f} \times \pi".format(-(alpha_string + beta_string) / 2),
    )
    Rz_C = Gate(
        "RZ",
        targets=[target],
        arg_value=(-alpha + beta) / 2,
        arg_label=r"{:0.2f} \times \pi".format((-alpha_string + beta_string) / 2),
    )

    return (Rz_A, Ry_A, Pauli_X, Ry_B, Rz_B, Pauli_X, Rz_C, Phase_gate)
