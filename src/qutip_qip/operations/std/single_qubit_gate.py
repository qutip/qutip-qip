import numpy as np

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
from qutip_qip.operations import Gate, AngleParametricGate


class _SingleQubitGate(Gate):
    """Abstract one-qubit gate."""
    num_qubits: int = 1

class X(_SingleQubitGate):
    """
    Single-qubit X gate.

    Examples
    --------
    >>> from qutip_qip.operations import X
    >>> X.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0. 1.]
     [1. 0.]]
    """

    latex_str = r"X"

    @staticmethod
    def get_qobj():
        return sigmax(dtype="dense")


class Y(_SingleQubitGate):
    """
    Single-qubit Y gate.

    Examples
    --------
    >>> from qutip_qip.operations import Y
    >>> Y(0).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0.+0.j 0.-1.j]
     [0.+1.j 0.+0.j]]
    """

    latex_str = r"Y"

    @staticmethod
    def get_qobj():
        return sigmay(dtype="dense")


class Z(_SingleQubitGate):
    """
    Single-qubit Z gate.

    Examples
    --------
    >>> from qutip_qip.operations import Z
    >>> Z(0).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]
    """

    latex_str = r"Z"

    @staticmethod
    def get_qobj():
        return sigmaz(dtype="dense")


class RX(AngleParametricGate):
    """
    Single-qubit rotation RX.

    Examples
    --------
    >>> from qutip_qip.operations import RX
    >>> RX(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     -0.70711j]
     [0.     -0.70711j 0.70711+0.j     ]]
    """
    num_qubits = 1
    num_params = 1
    latex_str = r"R_x"

    def get_qobj(self):
        phi = self.arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -1j * np.sin(phi / 2)],
                [-1j * np.sin(phi / 2), np.cos(phi / 2)],
            ],
            dims = [[2], [2]]
        )


class RY(AngleParametricGate):
    """
    Single-qubit rotation RY.

    Examples
    --------
    >>> from qutip_qip.operations import RY
    >>> RY(0, 3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    num_qubits = 1
    num_params: int = 1
    latex_str = r"R_y"

    def get_qobj(self):
        phi = self.arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -np.sin(phi / 2)],
                [np.sin(phi / 2), np.cos(phi / 2)],
            ]
        )


class RZ(AngleParametricGate):
    """
    Single-qubit rotation RZ.

    Examples
    --------
    >>> from qutip_qip.operations import RZ
    >>> RZ(0, 3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711-0.70711j 0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    num_qubits = 1
    num_params: int = 1
    latex_str = r"R_z"

    def get_qobj(self):
        phi = self.arg_value[0]
        return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])


class PHASE(AngleParametricGate):
    """
    PHASE Gate.

    Examples
    --------
    >>> from qutip_qip.operations import PHASE
    """

    num_qubits = 1
    num_params: int = 1
    latex_str = r"PHASE"

    def get_qobj(self):
        phi = self.arg_value[0]
        return Qobj(
            [
                [1, 0],
                [0, np.exp(1j * phi)],
            ]
        )


class IDLE(_SingleQubitGate):
    """
    IDLE gate.

    Examples
    --------
    >>> from qutip_qip.operations import IDLE
    """

    latex_str = r"{\rm IDLE}"

    @staticmethod
    def get_qobj():
        return qeye(2)


class H(_SingleQubitGate):
    """
    Hadamard gate.

    Examples
    --------
    >>> from qutip_qip.operations import H
    >>> H(0).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70711  0.70711]
     [ 0.70711 -0.70711]]
    """

    latex_str = r"H"

    @staticmethod
    def get_qobj():
        return 1 / np.sqrt(2.0) * Qobj([[1, 1], [1, -1]])


class SNOT(H):
    pass


class SQRTNOT(_SingleQubitGate):
    r"""
    :math:`\sqrt{X}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import SQRTNOT
    >>> SQRTNOT(0).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.5+0.5j 0.5-0.5j]
     [0.5-0.5j 0.5+0.5j]]
    """

    latex_str = r"\sqrt{\rm NOT}"

    @staticmethod
    def get_qobj():
        return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])


class S(_SingleQubitGate):
    r"""
    S gate or :math:`\sqrt{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import S
    >>> S(0).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+1.j]]
    """

    latex_str = r"{\rm S}"

    @staticmethod
    def get_qobj():
        return Qobj([[1, 0], [0, 1j]])


class T(_SingleQubitGate):
    r"""
    T gate or :math:`\sqrt[4]{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import T
    >>> T(0).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    latex_str = r"{\rm T}"

    @staticmethod
    def get_qobj():
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])


class R(AngleParametricGate):
    r"""
    Arbitrary single-qubit rotation

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & -ie^{-i\phi} \sin(\frac{\theta}{2}) \\
        -ie^{i\phi} \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import R
    >>> R(0, (np.pi/2, np.pi/2)).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    num_qubits = 1
    num_params: int = 2
    latex_str = r"{\rm R}"

    def get_qobj(self):
        phi, theta = self.arg_value
        return Qobj(
            [
                [
                    np.cos(theta / 2.0),
                    -1.0j * np.exp(-1.0j * phi) * np.sin(theta / 2.0),
                ],
                [
                    -1.0j * np.exp(1.0j * phi) * np.sin(theta / 2.0),
                    np.cos(theta / 2.0),
                ],
            ]
        )


class QASMU(AngleParametricGate):
    r"""
    QASMU gate.

    .. math::
        U(\theta, \phi, \gamma) = RZ(\phi) RY(\theta) RZ(\gamma)

    Examples
    --------
    >>> from qutip_qip.operations import QASMU
    >>> QASMU(0, (np.pi/2, np.pi, np.pi/2)).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[-0.5-0.5j -0.5+0.5j]
     [ 0.5+0.5j -0.5+0.5j]]
    """

    num_qubits = 1
    num_params: int = 3
    latex_str = r"{\rm QASMU}"

    def get_qobj(self):
        theta, phi, gamma = self.arg_value
        return Qobj(
            [
                [
                    np.exp(-1j * (phi + gamma) / 2) * np.cos(theta / 2),
                    -np.exp(-1j * (phi - gamma) / 2) * np.sin(theta / 2),
                ],
                [
                    np.exp(1j * (phi - gamma) / 2) * np.sin(theta / 2),
                    np.exp(1j * (phi + gamma) / 2) * np.cos(theta / 2),
                ],
            ]
        )
