import warnings
import numpy as np

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
from qutip_qip.operations import Gate, AngleParametricGate


class _SingleQubitGate(Gate):
    """Abstract one-qubit gate."""

    __slots__ = ()
    num_qubits = 1


class _SingleQubitParametricGate(AngleParametricGate):
    """Abstract one-qubit parametric gate."""

    __slots__ = ()
    num_qubits = 1


class X(_SingleQubitGate):
    """
    Single-qubit X gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import X
    >>> X.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0. 1.]
     [1. 0.]]
    """

    __slots__ = ()

    self_inverse = True
    is_clifford = True
    latex_str = r"X"

    @staticmethod
    def get_qobj() -> Qobj:
        return sigmax(dtype="dense")


class Y(_SingleQubitGate):
    """
    Single-qubit Y gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import Y
    >>> Y.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0.+0.j 0.-1.j]
     [0.+1.j 0.+0.j]]
    """

    __slots__ = ()

    self_inverse = True
    is_clifford = True
    latex_str = r"Y"

    @staticmethod
    def get_qobj() -> Qobj:
        return sigmay(dtype="dense")


class Z(_SingleQubitGate):
    """
    Single-qubit Z gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import Z
    >>> Z.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]
    """

    __slots__ = ()

    self_inverse = True
    is_clifford = True
    latex_str = r"Z"

    @staticmethod
    def get_qobj() -> Qobj:
        return sigmaz(dtype="dense")


class IDLE(_SingleQubitGate):
    """
    IDLE gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import IDLE
    """

    __slots__ = ()

    self_inverse = True
    is_clifford = True
    latex_str = r"{\rm IDLE}"

    @staticmethod
    def get_qobj() -> Qobj:
        return qeye(2)


class H(_SingleQubitGate):
    """
    Hadamard gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import H
    >>> H.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70711  0.70711]
     [ 0.70711 -0.70711]]
    """

    __slots__ = ()

    self_inverse = True
    is_clifford = True
    latex_str = r"H"

    @staticmethod
    def get_qobj() -> Qobj:
        return 1 / np.sqrt(2.0) * Qobj([[1, 1], [1, -1]])


class SNOT(H):
    """
    Hadamard gate (Deprecated, use H instead).
    """

    __slots__ = ()

    def __init__(self):
        warnings.warn(
            "SNOT is deprecated and will be removed in future versions. "
            "Use H instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()


class SQRTX(_SingleQubitGate):
    r"""
    :math:`\sqrt{X}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SQRTX
    >>> SQRTX.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.5+0.5j 0.5-0.5j]
     [0.5-0.5j 0.5+0.5j]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"\sqrt{\rm X}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])

    @staticmethod
    def inverse_gate() -> Gate:
        return SQRTXdag


class SQRTXdag(_SingleQubitGate):
    r"""
    :math:`\sqrt{X}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SQRTXdag
    >>> SQRTXdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.5-0.5j 0.5+0.5j]
     [0.5+0.5j 0.5-0.5j]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"\sqrt{\rm X}^\dagger"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]])

    @staticmethod
    def inverse_gate() -> Gate:
        return SQRTX


class SQRTNOT(SQRTX):
    __slots__ = ()

    def __init__(self):
        warnings.warn(
            "SQRTNOT is deprecated and will be removed in future versions. "
            "Use SQRTX instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()


class S(_SingleQubitGate):
    r"""
    S gate or :math:`\sqrt{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import S
    >>> S.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+1.j]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"{\rm S}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, 1j]])

    @staticmethod
    def inverse_gate() -> Gate:
        return Sdag


class Sdag(_SingleQubitGate):
    r"""
    S gate or :math:`\sqrt{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import S
    >>> Sdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.-1.j]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"{\rm S^\dagger}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, -1j]])

    @staticmethod
    def inverse_gate() -> Gate:
        return S


class T(_SingleQubitGate):
    r"""
    T gate or :math:`\sqrt[4]{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import T
    >>> T.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    __slots__ = ()

    self_inverse = False
    latex_str = r"{\rm T}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    @staticmethod
    def inverse_gate() -> Gate:
        return Tdag


class Tdag(_SingleQubitGate):
    r"""
    Tdag gate or :math:`\sqrt[4]{Z}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import Tdag
    >>> Tdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711-0.70711j]]
    """

    __slots__ = ()

    self_inverse = False
    latex_str = r"{\rm Tdag}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, np.exp(-1j * np.pi / 4)]])

    @staticmethod
    def inverse_gate() -> Gate:
        return T


class RX(_SingleQubitParametricGate):
    """
    Single-qubit rotation RX.

    Examples
    --------
    >>> from qutip_qip.operations.std import RX
    >>> RX(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     -0.70711j]
     [0.     -0.70711j 0.70711+0.j     ]]
    """

    __slots__ = ()

    num_params = 1
    latex_str = r"R_x"

    def get_qobj(self) -> Qobj:
        phi = self.arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -1j * np.sin(phi / 2)],
                [-1j * np.sin(phi / 2), np.cos(phi / 2)],
            ],
            dims=[[2], [2]],
        )

    def inverse_gate(self) -> Gate:
        theta = self.arg_value[0]
        return RX(-theta)


class RY(_SingleQubitParametricGate):
    """
    Single-qubit rotation RY.

    Examples
    --------
    >>> from qutip_qip.operations.std import RY
    >>> RY(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    __slots__ = ()

    num_params = 1
    latex_str = r"R_y"

    def get_qobj(self) -> Qobj:
        phi = self.arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -np.sin(phi / 2)],
                [np.sin(phi / 2), np.cos(phi / 2)],
            ]
        )

    def inverse_gate(self) -> Gate:
        theta = self.arg_value[0]
        return RY(-theta)


class RZ(_SingleQubitParametricGate):
    """
    Single-qubit rotation RZ.

    Examples
    --------
    >>> from qutip_qip.operations.std import RZ
    >>> RZ(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711-0.70711j 0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    __slots__ = ()

    num_params = 1
    latex_str = r"R_z"

    def get_qobj(self) -> Qobj:
        phi = self.arg_value[0]
        return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])

    def inverse_gate(self) -> Gate:
        theta = self.arg_value[0]
        return RZ(-theta)


class PHASE(_SingleQubitParametricGate):
    """
    PHASE Gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import PHASE
    """

    __slots__ = ()

    num_params = 1
    latex_str = r"PHASE"

    def get_qobj(self) -> Qobj:
        phi = self.arg_value[0]
        return Qobj(
            [
                [1, 0],
                [0, np.exp(1j * phi)],
            ]
        )

    def inverse_gate(self) -> Gate:
        phi = self.arg_value[0]
        return PHASE(-phi)


class R(_SingleQubitParametricGate):
    r"""
    Arbitrary single-qubit rotation

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & -ie^{-i\phi} \sin(\frac{\theta}{2}) \\
        -ie^{i\phi} \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations.std import R
    >>> R((np.pi/2, np.pi/2)).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    __slots__ = ()

    num_params = 2
    latex_str = r"{\rm R}"

    def get_qobj(self) -> Qobj:
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

    def inverse_gate(self) -> Gate:
        phi, theta = self.arg_value
        return R([phi, -theta])


class QASMU(_SingleQubitParametricGate):
    r"""
    QASMU gate.

    .. math::
        U(\theta, \phi, \gamma) = RZ(\phi) RY(\theta) RZ(\gamma)

    Examples
    --------
    >>> from qutip_qip.operations.std import QASMU
    >>> QASMU(0, (np.pi/2, np.pi, np.pi/2)).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[-0.5-0.5j -0.5+0.5j]
     [ 0.5+0.5j -0.5+0.5j]]
    """

    __slots__ = ()

    num_params = 3
    latex_str = r"{\rm QASMU}"

    def get_qobj(self) -> Qobj:
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

    def inverse_gate(self) -> Gate:
        theta, phi, gamma = self.arg_value
        return QASMU([-theta, -gamma, -phi])
