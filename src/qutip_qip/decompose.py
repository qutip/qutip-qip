import numpy as np
import cmath

from qutip import Qobj
from qutip_qip._decomposition_functions._utility import (
    check_gate,
    MethodError,
    GateError,
)

from qutip_qip.circuit import QubitCircuit, Gate

from qutip_qip._decomposition_functions._single_qubit_gate import (
    _ZYZ_rotation,
    _ZXZ_rotation,
    _ZYZ_pauli_X,
)

_single_decompositions_dictionary = {
    "ZYZ": _ZYZ_rotation,
    "ZXZ": _ZXZ_rotation,
    "ZYZ_PauliX": _ZYZ_pauli_X,
}  # other combinations to add here


def decompose_one_qubit_gate(input_gate, method, num_qubits, target=0):
    r""" An input 1-qubit gate is expressed as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j` or as a product of rotation matrices
    :math:`\textrm{R}_i` and :math:`\textrm{R}_j` and a Pauli :math:`\sigma_k`.

    Here, :math:`i \neq j` and :math:`i, j, k \in {x, y, z}`.

    Based on Lemma 4.1 and Lemma 4.3 of https://arxiv.org/abs/quant-ph/9503016v1 respectively.

    .. math::

        U = \begin{bmatrix}
            a       & b \\
            -b^*       & a^*  \\
            \end{bmatrix} = \textrm{R}_i(\alpha)  \textrm{R}_j(\theta)  \textrm{R}_i(\beta) = \textrm{A} \sigma_k \textrm{B} \sigma_k \textrm{C}

    Here,

    * :math:`\textrm{A} = \textrm{R}_i(\alpha) \textrm{R}_j \left(\frac{\theta}{2} \right)`

    * :math:`\textrm{B} = \textrm{R}_j \left(\frac{-\theta}{2} \right) \textrm{R}_i \left(\frac{- \left(\alpha + \beta \right)}{2} \right)`

    * :math:`\textrm{C} = \textrm{R}_i \left(\frac{\left(-\alpha + \beta \right)}{2} \right)`


    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
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

        When the input gate is decomposed to product of rotation matrices - tuple
        will contain 4 elements per each :math:`1 \times 1`
        qubit gate - :math:`\textrm{R}_i(\alpha)`, :math:`\textrm{R}_j(\theta)`,
        :math:`\textrm{R}_i(\beta)`, and some global phase gate.

        When the input gate is decomposed to product of rotation matrices and Pauli -
        tuple will contain 6 elements per each :math:`1 \times 1`
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

    key = _single_decompositions_dictionary.keys()
    if str(method) in key:
        method = _single_decompositions_dictionary[str(method)]
        return method(input_gate, target, 1)

    else:
        raise MethodError("Invalid method chosen.")
