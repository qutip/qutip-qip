import numpy as np
import cmath


from qutip_qip._decompose._utility import (
    check_gate,
    MethodError,
)

from qutip_qip.circuit import Gate


__all__ = ["decompose_one_qubit_gate"]


def _angles_for_ZYZ(input_gate):
    """Finds and returns the angles for ZYZ rotation matrix. These are
    used to change ZYZ to other combinations.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The gate matrix that's supposed to be decomposed should be a Qobj.
    """
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


def _ZYZ_rotation(input_gate):
    r"""An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_y`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    check_gate(input_gate, num_qubits=1)
    alpha, theta, beta, global_phase_angle = _angles_for_ZYZ(input_gate)

    Phase_gate = Gate(
        "GLOBALPHASE",
        targets=[0],
        arg_value=global_phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(global_phase_angle / np.pi),
    )
    Rz_beta = Gate(
        "RZ",
        targets=[0],
        arg_value=beta,
        arg_label=r"{:0.2f} \times \pi".format(beta / np.pi),
    )
    Ry_theta = Gate(
        "RY",
        targets=[0],
        arg_value=theta,
        arg_label=r"{:0.2f} \times \pi".format(theta / np.pi),
    )
    Rz_alpha = Gate(
        "RZ",
        targets=[0],
        arg_value=alpha,
        arg_label=r"{:0.2f} \times \pi".format(alpha / np.pi),
    )

    return (Rz_alpha, Ry_theta, Rz_beta, Phase_gate)


def _ZXZ_rotation(input_gate):
    r"""An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_z` and :math:`\textrm{R}_x`.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    check_gate(input_gate, num_qubits=1)
    alpha, theta, beta, global_phase_angle = _angles_for_ZYZ(input_gate)
    alpha = alpha - np.pi / 2
    beta = beta + np.pi / 2
    # theta and global phase are same as ZYZ values

    Phase_gate = Gate(
        "GLOBALPHASE",
        targets=[0],
        arg_value=global_phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(global_phase_angle / np.pi),
    )
    Rz_alpha = Gate(
        "RZ",
        targets=[0],
        arg_value=alpha,
        arg_label=r"{:0.2f} \times \pi".format(alpha / np.pi),
    )
    Rx_theta = Gate(
        "RX",
        targets=[0],
        arg_value=theta,
        arg_label=r"{:0.2f} \times \pi".format(theta / np.pi),
    )
    Rz_beta = Gate(
        "RZ",
        targets=[0],
        arg_value=beta,
        arg_label=r"{:0.2f} \times \pi".format(beta / np.pi),
    )

    return (Rz_alpha, Rx_theta, Rz_beta, Phase_gate)


# Functions for ABC_decomposition


def _ZYZ_pauli_X(input_gate):
    """Returns a 1 qubit unitary as a product of ZYZ rotation matrices and
    Pauli X."""
    check_gate(input_gate, num_qubits=1)
    alpha, theta, beta, global_phase_angle = _angles_for_ZYZ(input_gate)

    Phase_gate = Gate(
        "GLOBALPHASE",
        targets=[0],
        arg_value=global_phase_angle,
        arg_label=r"{:0.2f} \times \pi".format(global_phase_angle / np.pi),
    )
    Rz_A = Gate(
        "RZ",
        targets=[0],
        arg_value=alpha,
        arg_label=r"{:0.2f} \times \pi".format(alpha / np.pi),
    )
    Ry_A = Gate(
        "RY",
        targets=[0],
        arg_value=theta / 2,
        arg_label=r"{:0.2f} \times \pi".format(theta / np.pi),
    )
    Pauli_X = Gate("X", targets=[0])
    Ry_B = Gate(
        "RY",
        targets=[0],
        arg_value=-theta / 2,
        arg_label=r"{:0.2f} \times \pi".format(-theta / np.pi),
    )
    Rz_B = Gate(
        "RZ",
        targets=[0],
        arg_value=-(alpha + beta) / 2,
        arg_label=r"{:0.2f} \times \pi".format(-(alpha + beta) / (2 * np.pi)),
    )
    Rz_C = Gate(
        "RZ",
        targets=[0],
        arg_value=(-alpha + beta) / 2,
        arg_label=r"{:0.2f} \times \pi".format((-alpha + beta) / (2 * np.pi)),
    )

    return (Rz_A, Ry_A, Pauli_X, Ry_B, Rz_B, Pauli_X, Rz_C, Phase_gate)


_single_decompositions_dictionary = {
    "ZYZ": _ZYZ_rotation,
    "ZXZ": _ZXZ_rotation,
    "ZYZ_PauliX": _ZYZ_pauli_X,
}  # other combinations to add here


def decompose_one_qubit_gate(input_gate, method):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j` or as a product of rotation
    matrices :math:`\textrm{R}_i` and :math:`\textrm{R}_j` and a
    Pauli :math:`\sigma_k`.

    Here, :math:`i \neq j` and :math:`i, j, k \in {x, y, z}`.

    Based on Lemma 4.1 and Lemma 4.3 of
    https://arxiv.org/abs/quant-ph/9503016v1 respectively.

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{R}_i(\alpha)  \textrm{R}_j(\theta) \textrm{R}_i(\beta) = \textrm{A} \sigma_k \textrm{B} \sigma_k \textrm{C}

    Here,

    * :math:`\textrm{A} = \textrm{R}_i(\alpha) \textrm{R}_j\left(\frac{\theta}{2} \right)`

    * :math:`\textrm{B} = \textrm{R}_j \left(\frac{-\theta}{2} \right)\textrm{R}_i \left(\frac{- \left(\alpha + \beta \right)}{2} \right)`

    * :math:`\textrm{C} = \textrm{R}_i \left(\frac{\left(-\alpha + \beta\right)}{2} \right)`


    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix to be decomposed.


    method : string
        Name of the preferred decomposition method

        .. list-table::
            :widths: auto
            :header-rows: 1

            * - Method Key
              - Method
            * - ZYZ
              - :math:`\textrm{R}_z(\alpha)  \textrm{R}_y(\theta) \textrm{R}_z(\beta)`
            * - ZXZ
              - :math:`\textrm{R}_z(\alpha)  \textrm{R}_x(\theta) \textrm{R}_z(\beta)`
            * - ZYZ_PauliX
              - :math:`\textrm{A} \sigma_k \textrm{B} \sigma_k \textrm{C}` :math:`\forall k =x, i =z, j=y`


        .. note::
            This function is under construction. As more combinations are
            added, above table will be updated with their respective keys.


    Returns
    -------
    tuple
        The gates in the decomposition are returned as a tuple of :class:`Gate`
        objects.

        When the input gate is decomposed to product of rotation matrices -
        tuple will contain 4 elements per each :math:`1 \times 1`
        qubit gate - :math:`\textrm{R}_i(\alpha)`, :math:`\textrm{R}_j(\theta)`
        , :math:`\textrm{R}_i(\beta)`, and some global phase gate.

        When the input gate is decomposed to product of rotation matrices and
        Pauli - tuple will contain 6 elements per each :math:`1 \times 1`
        qubit gate - 2 gates forming :math:`\textrm{A}`, 2 gates forming
        :math:`\textrm{B}`, 1 gates forming :math:`\textrm{C}`, and some global
        phase gate.
    """
    check_gate(input_gate, num_qubits=1)
    f = _single_decompositions_dictionary.get(method, None)
    if f is None:
        raise MethodError(f"Invalid decomposition method: {method!r}")
    return f(input_gate)
