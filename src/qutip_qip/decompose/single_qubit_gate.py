import numpy as np
import cmath

from qutip import Qobj
from qutip_qip.decompose._utility import check_gate, MethodError, GateError

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


_rotation_matrices_dictionary = {
    "ZYZ": _ZYZ_rotation,
    "ZXZ": _ZXZ_rotation,
}  # other combinations to add here


def decompose_to_rotation_matrices(input_gate, method, num_qubits, target=0):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j`.

    Here, :math:`i \neq j` and :math:`i, j \in {x, y, z}`.

    Based on Lemma 4.1 of https://arxiv.org/abs/quant-ph/9503016v1

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{R}_i(\alpha)  \textrm{R}_j(\theta)  \textrm{R}_i(\beta)

    Parameters
    ----------
    input_gate : :class:`.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    num_qubits : int
        Number of qubits being acted upon by input gate

    target : int
        If the circuit contains more than 1 qubits then provide target for
        single qubit gate.

    method : string
        Name of the preferred decomposition method

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
        assert num_qubits == 1
    except AssertionError:
        if target is None and num_qubits > 1:
            raise GateError(
                "This method is valid for single qubit gates only. Provide a target qubit for single qubit gate."
            )

    key = _rotation_matrices_dictionary.keys()
    if str(method) in key:
        method = _rotation_matrices_dictionary[str(method)]
        return method(input_gate, target, 1)

    else:
        raise MethodError("Invalid method chosen.")


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


_rotation_pauli_matrices_dictionary = {
    "ZYZ_PauliX": _ZYZ_pauli_X,
}  # other combinations to add here


def ABC_decomposition(input_gate, method, num_qubits, target=0):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j` and Pauli :math:`\sigma_k`.

    Here, :math:`i \neq j` and :math:`i, j, k \in {x, y, z}`.

    Based on Lemma 4.3 of https://arxiv.org/abs/quant-ph/9503016v1

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{A} \sigma_k \textrm{B} \sigma_k \textrm{C}

    Here,

    .. list-table::
        :widths: auto
        :header-rows: 1

        * - Gate Label
          - Gate Composition
        * - :math:`\textrm{A}`
          - :math:`\textrm{R}_i(\alpha) \textrm{R}_j \left(\frac{\theta}{2} \right)`
        * - :math:`\textrm{B}`
          - :math:`\textrm{R}_j \left(\frac{-\theta}{2} \right) \textrm{R}_i \left(\frac{- \left(\alpha + \beta \right)}{2} \right)`
        * - :math:`\textrm{C}`
          - :math:`\textrm{R}_i \left(\frac{\left(-\alpha + \beta \right)}{2} \right)`

    Parameters
    ----------
    input_gate : :class:`.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    num_qubits : int
        Number of qubits being acted upon by input gate

    target : int
        If the circuit contains more than 1 qubits then provide target for
        single qubit gate.

    method : string
        Name of the preferred decomposition method

            .. list-table::
                :widths: auto
                :header-rows: 1

                * - Method Key
                  - Method
                  - :math:`(i,j,k)`
                * - ZYZ_PauliX
                  - :math:`\textrm{A} \textrm{X} \textrm{B} \textrm{X} \textrm{C}`
                  - :math:`(z,y,x)`

    Returns
    -------
    tuple
        The gates in the decomposition are returned as a tuple of :class:`Gate`
        objects. This tuple will contain 6 elements per each :math:`1 \times 1`
        qubit gate - 2 gates forming :math:`\textrm{A}`, 2 gates forming :math:`\textrm{B}`,
        1 gates forming :math:`\textrm{C}`, and some global phase gate.
    """
    try:
        assert num_qubits == 1
    except AssertionError:
        if target is None and num_qubits > 1:
            raise GateError(
                "This method is valid for single qubit gates only. Provide a target qubit for single qubit gate."
            )

    key = _rotation_pauli_matrices_dictionary.keys()
    if str(method) in key:
        method = _rotation_pauli_matrices_dictionary[str(method)]
        return method(input_gate, target, 1)

    else:
        raise MethodError("Invalid method chosen.")
