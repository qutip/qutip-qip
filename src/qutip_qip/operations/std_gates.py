import numpy as np
import scipy.sparse as sp

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
from qutip_qip.operations import (
    SingleQubitGate,
    TwoQubitGate,
    ControlledGate,
    ParametrizedGate,
    ControlledParamGate,
    ParametrizedSingleQubitGate,
    ParametrizedTwoQubitGate,
)

######################### Single Qubit Gates ############################


class X(SingleQubitGate):
    """
    Single-qubit X gate.

    Examples
    --------
    >>> from qutip_qip.operations import X
    >>> X(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0. 1.]
     [1. 0.]]
    """

    latex_str = r"X"

    @staticmethod
    def get_compact_qobj():
        return sigmax(dtype="dense")


class Y(SingleQubitGate):
    """
    Single-qubit Y gate.

    Examples
    --------
    >>> from qutip_qip.operations import Y
    >>> Y(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0.+0.j 0.-1.j]
     [0.+1.j 0.+0.j]]
    """

    latex_str = r"Y"

    @staticmethod
    def get_compact_qobj():
        return sigmay(dtype="dense")


class Z(SingleQubitGate):
    """
    Single-qubit Z gate.

    Examples
    --------
    >>> from qutip_qip.operations import Z
    >>> Z(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]
    """

    latex_str = r"Z"

    @staticmethod
    def get_compact_qobj():
        return sigmaz(dtype="dense")


class PHASE(ParametrizedSingleQubitGate):
    """
    PHASE Gate.

    Examples
    --------
    >>> from qutip_qip.operations import PHASE
    """

    latex_str = r"PHASE"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj(
            [
                [1, 0],
                [0, np.exp(1j*phi)],
            ]
        )


class RX(ParametrizedSingleQubitGate):
    """
    Single-qubit rotation RX.

    Examples
    --------
    >>> from qutip_qip.operations import RX
    >>> RX(0, 3.14159/2).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     -0.70711j]
     [0.     -0.70711j 0.70711+0.j     ]]
    """

    latex_str = r"R_x"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj(
            [
                [np.cos(phi / 2), -1j * np.sin(phi / 2)],
                [-1j * np.sin(phi / 2), np.cos(phi / 2)],
            ]
        )


class RY(ParametrizedSingleQubitGate):
    """
    Single-qubit rotation RY.

    Examples
    --------
    >>> from qutip_qip.operations import RY
    >>> RY(0, 3.14159/2).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    latex_str = r"R_y"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj(
            [
                [np.cos(phi / 2), -np.sin(phi / 2)],
                [np.sin(phi / 2), np.cos(phi / 2)],
            ]
        )


class RZ(ParametrizedSingleQubitGate):
    """
    Single-qubit rotation RZ.

    Examples
    --------
    >>> from qutip_qip.operations import RZ
    >>> RZ(0, 3.14159/2).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711-0.70711j 0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    latex_str = r"R_z"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])


class IDLE(SingleQubitGate):
    """
    IDLE gate.

    Examples
    --------
    >>> from qutip_qip.operations import IDLE
    """

    latex_str = r"{\rm IDLE}"

    @staticmethod
    def get_compact_qobj():
        return qeye(2)


class H(SingleQubitGate):
    """
    Hadamard gate.

    Examples
    --------
    >>> from qutip_qip.operations import H
    >>> H(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70711  0.70711]
     [ 0.70711 -0.70711]]
    """

    latex_str = r"H"

    @staticmethod
    def get_compact_qobj():
        return 1 / np.sqrt(2.0) * Qobj([[1, 1], [1, -1]])


class SNOT(H):
    pass


class SQRTNOT(SingleQubitGate):
    r"""
    :math:`\sqrt{X}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import SQRTNOT
    >>> SQRTNOT(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.5+0.5j 0.5-0.5j]
     [0.5-0.5j 0.5+0.5j]]
    """

    latex_str = r"\sqrt{\rm NOT}"

    @staticmethod
    def get_compact_qobj():
        return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])


class S(SingleQubitGate):
    r"""
    S gate or :math:`\sqrt{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import S
    >>> S(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+1.j]]
    """

    latex_str = r"{\rm S}"

    @staticmethod
    def get_compact_qobj():
        return Qobj([[1, 0], [0, 1j]])


class T(SingleQubitGate):
    r"""
    T gate or :math:`\sqrt[4]{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import T
    >>> T(0).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    latex_str = r"{\rm T}"

    @staticmethod
    def get_compact_qobj():
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])


class R(ParametrizedSingleQubitGate):
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
    >>> R(0, (np.pi/2, np.pi/2)).get_compact_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    latex_str = r"{\rm R}"

    def get_compact_qobj(self):
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


class QASMU(ParametrizedSingleQubitGate):
    r"""
    QASMU gate.

    .. math::
        U(\theta, \phi, \gamma) = RZ(\phi) RY(\theta) RZ(\gamma)

    Examples
    --------
    >>> from qutip_qip.operations import QASMU
    >>> QASMU(0, (np.pi/2, np.pi, np.pi/2)).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[-0.5-0.5j -0.5+0.5j]
     [ 0.5+0.5j -0.5+0.5j]]
    """

    latex_str = r"{\rm QASM-U}"

    def get_compact_qobj(self):
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


############################ Two Qubit Gates #########################


class SWAP(TwoQubitGate):
    """
    SWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations import SWAP
    >>> SWAP([0, 1]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]
    """

    latex_str = r"{\rm SWAP}"

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )


class ISWAP(TwoQubitGate):
    """
    iSWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations import ISWAP
    >>> ISWAP([0, 1]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+1.j 0.+0.j]
     [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """

    latex_str = r"{i}{\rm SWAP}"

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )


class SQRTSWAP(TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{SWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import SQRTSWAP
    >>> SQRTSWAP([0, 1]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    latex_str = r"\sqrt{\rm SWAP}"

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                    [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                    [0, 0, 0, 1],
                ]
            ),
            dims=[[2, 2], [2, 2]],
        )


class SQRTISWAP(TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{iSWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import SQRTISWAP
    >>> SQRTISWAP([0, 1]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j      0.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.j      0.     +0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     +0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     +0.j      0.     +0.j      0.     +0.j      1.     +0.j     ]]
    """

    latex_str = r"\sqrt{{i}\rm SWAP}"

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                    [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 0, 0, 1],
                ]
            ),
            dims=[[2, 2], [2, 2]],
        )


class BERKELEY(TwoQubitGate):
    r"""
    BERKELEY gate.

    .. math::

        \begin{pmatrix}
        \cos(\frac{\pi}{8}) & 0 & 0 & i\sin(\frac{\pi}{8}) \\
        0 & \cos(\frac{3\pi}{8}) & i\sin(\frac{3\pi}{8}) & 0 \\
        0 & i\sin(\frac{3\pi}{8}) & \cos(\frac{3\pi}{8}) & 0 \\
        i\sin(\frac{\pi}{8}) & 0 & 0 & \cos(\frac{\pi}{8})
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import BERKELEY
    >>> BERKELEY([0, 1]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92388+0.j      0.     +0.j      0.     +0.j      0.     +0.38268j]
     [0.     +0.j      0.38268+0.j      0.     +0.92388j 0.     +0.j     ]
     [0.     +0.j      0.     +0.92388j 0.38268+0.j      0.     +0.j     ]
     [0.     +0.38268j 0.     +0.j      0.     +0.j      0.92388+0.j     ]]
    """

    latex_str = r"{\rm BERKELEY}"

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [
                [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
                [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
                [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
            ],
            dims=[[2, 2], [2, 2]],
        )


class SWAPALPHA(ParametrizedTwoQubitGate):
    r"""
    SWAPALPHA gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & \frac{1 + e^{i\pi\alpha}}{2} & \frac{1 - e^{i\pi\alpha}}{2} & 0 \\
        0 & \frac{1 - e^{i\pi\alpha}}{2} & \frac{1 + e^{i\pi\alpha}}{2} & 0 \\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import SWAPALPHA
    >>> SWAPALPHA([0, 1], 0.5).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    latex_str = r"{\rm SWAPALPHA}"

    def get_compact_qobj(self):
        alpha = self.arg_value
        return Qobj(
            [
                [1, 0, 0, 0],
                [
                    0,
                    0.5 * (1 + np.exp(1.0j * np.pi * alpha)),
                    0.5 * (1 - np.exp(1.0j * np.pi * alpha)),
                    0,
                ],
                [
                    0,
                    0.5 * (1 - np.exp(1.0j * np.pi * alpha)),
                    0.5 * (1 + np.exp(1.0j * np.pi * alpha)),
                    0,
                ],
                [0, 0, 0, 1],
            ],
            dims=[[2, 2], [2, 2]],
        )


class MS(ParametrizedTwoQubitGate):
    r"""
    Mølmer–Sørensen gate.

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & 0 & 0 & -ie^{-i2\phi}\sin(\frac{\theta}{2}) \\
        0 & \cos(\frac{\theta}{2}) & -i\sin(\frac{\theta}{2}) & 0 \\
        0 & -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) & 0 \\
        -ie^{i2\phi}\sin(\frac{\theta}{2}) & 0 & 0 & \cos(\frac{\theta}{2})
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import MS
    >>> MS([0, 1], (np.pi/2, 0)).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     +0.j      0.     +0.j      0.     -0.70711j]
     [0.     +0.j      0.70711+0.j      0.     -0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     -0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     -0.70711j 0.     +0.j      0.     +0.j      0.70711+0.j     ]]
    """

    latex_str = r"{\rm MS}"

    def get_compact_qobj(self):
        theta, phi = self.arg_value
        return Qobj(
            [
                [
                    np.cos(theta / 2),
                    0,
                    0,
                    -1j * np.exp(-1j * 2 * phi) * np.sin(theta / 2),
                ],
                [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
                [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
                [
                    -1j * np.exp(1j * 2 * phi) * np.sin(theta / 2),
                    0,
                    0,
                    np.cos(theta / 2),
                ],
            ],
            dims=[[2, 2], [2, 2]],
        )


class RZX(ParametrizedTwoQubitGate):
    r"""
    RZX gate.

    .. math::

        \begin{pmatrix}
        \cos{\theta/2} & -i\sin{\theta/2} & 0 & 0 \\
        -i\sin{\theta/2} & \cos{\theta/2} & 0 & 0 \\
        0 & 0 & \cos{\theta/2} & i\sin{\theta/2} \\
        0 & 0 & i\sin{\theta/2} & \cos{\theta/2} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import RZX
    >>> RZX([0, 1], np.pi).get_compact_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.+0.j 0.-1.j 0.+0.j 0.+0.j]
    [0.-1.j 0.+0.j 0.+0.j 0.+0.j]
    [0.+0.j 0.+0.j 0.+0.j 0.+1.j]
    [0.+0.j 0.+0.j 0.+1.j 0.+0.j]]
    """

    latex_str = r"{\rm RZX}"

    def get_compact_qobj(self):
        theta = self.arg_value
        return Qobj(
            np.array(
                [
                    [np.cos(theta / 2), -1.0j * np.sin(theta / 2), 0.0, 0.0],
                    [-1.0j * np.sin(theta / 2), np.cos(theta / 2), 0.0, 0.0],
                    [0.0, 0.0, np.cos(theta / 2), 1.0j * np.sin(theta / 2)],
                    [0.0, 0.0, 1.0j * np.sin(theta / 2), np.cos(theta / 2)],
                ]
            ),
            dims=[[2, 2], [2, 2]],
        )


class _ControlledTwoQubitGate(ControlledGate, TwoQubitGate):
    """
    This class allows correctly generating the gate instance
    when a redundant control_value is given, e.g.
    ``CNOT(0, 1, control_value=1)``,
    and raise an error if it is 0.
    """


class CNOT(_ControlledTwoQubitGate):
    """
    CNOT gate.

    Examples
    --------
    >>> from qutip_qip.operations import CNOT
    >>> CNOT(0, 1).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]
    """

    latex_str = r"{\rm CNOT}"
    _target_gate_class = X

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dims=[[2, 2], [2, 2]],
        )


class CX(CNOT):
    pass


class CY(_ControlledTwoQubitGate):
    """
    Controlled CY gate.

    Examples
    --------
    >>> from qutip_qip.operations import CY
    >>> CSIGN(0, 1).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0j  0.+0j  0.+0j  0.+0j]
     [ 0.+0j  1.+0j  0.+0j  0.+0j]
     [ 0.+0j  0.+0j  0.+0j  0.-1j]
     [ 0+0j.  0.+0j  0.+1j. 0.+0j]]
    """

    latex_str = r"{\rm CY}"
    _target_gate_class = Y

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dims=[[2, 2], [2, 2]],
        )


class CZ(_ControlledTwoQubitGate):
    """
    Controlled Z gate. Identical to the CSIGN gate.

    Examples
    --------
    >>> from qutip_qip.operations import CZ
    >>> CSIGN(0, 1).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """

    latex_str = r"{\rm CZ}"
    _target_gate_class = Z

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dims=[[2, 2], [2, 2]],
        )


class CSIGN(CZ):
    pass


class CH(_ControlledTwoQubitGate):
    r"""
    CH gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import CH
    """

    latex_str = r"{\rm CH}"
    _target_gate_class = H

    @staticmethod
    def get_compact_qobj():
        sq_2 = 1 / np.sqrt(2)
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, sq_2, sq_2],
                [0, 0, sq_2, -sq_2],
            ],
            dims=[[2, 2], [2, 2]],
        )


class CT(_ControlledTwoQubitGate):
    r"""
    CT gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import CPHASE
    """

    latex_str = r"{\rm CT}"
    _target_gate_class = T

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, (1 + 1j) / np.sqrt(2)],
            ],
            dims=[[2, 2], [2, 2]],
        )


class CS(_ControlledTwoQubitGate):
    r"""
    CS gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import CPHASE
    """

    latex_str = r"{\rm CS}"
    _target_gate_class = S

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]
            ),
            dims=[[2, 2], [2, 2]],
        )


class _ControlledParamTwoQubitGate(ControlledParamGate):
    """
    This class allows correctly generating the gate instance
    when a redundant control_value is given, e.g.
    ``CNOT(0, 1, control_value=1)``,
    and raise an error if it is 0.
    """
    ...


class CPHASE(_ControlledParamTwoQubitGate):
    r"""
    CPHASE gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import CPHASE
    >>> CPHASE(0, 1, np.pi/2).get_compact_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    latex_str = r"{\rm CPHASE}"
    _target_gate_class = PHASE

    def get_compact_qobj(self):
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * self.arg_value)],
            ],
            dims=[[2, 2], [2, 2]],
        )


class CRX(_ControlledParamTwoQubitGate):
    r"""
    Controlled X rotation.

    Examples
    --------
    >>> from qutip_qip.operations import CRX
    """

    latex_str = r"{\rm CRX}"
    _target_gate_class = RX


class CRY(_ControlledParamTwoQubitGate):
    r"""
    Controlled Y rotation.

    Examples
    --------
    >>> from qutip_qip.operations import CRY
    """

    latex_str = r"{\rm CRY}"
    _target_gate_class = RY


class CRZ(_ControlledParamTwoQubitGate):
    r"""
    CRZ gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
        0 & 0 & 0 & e^{i\frac{\theta}{2}} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations import CRZ
    >>> CRZ(0, 1, np.pi).get_compact_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    latex_str = r"{\rm CRZ}"
    _target_gate_class = RZ


class CQASMU(_ControlledParamTwoQubitGate):
    r"""
    Controlled QASMU rotation.

    Examples
    --------
    >>> from qutip_qip.operations import CQASMU
    """

    latex_str = r"{\rm CQASMU}"
    _target_gate_class = QASMU



########################### Special Gates #########################


class GLOBALPHASE(ParametrizedGate):
    """
    GLOBALPHASE gate.

    Examples
    --------
    >>> from qutip_qip.operations import GLOBALPHASE
    """

    latex_str = r"{\rm GLOBALPHASE}"

    def __init__(
        self,
        arg_value: float,
        arg_label: str = None,
        classical_controls=None,
        classical_control_value=None,
        style=None,
    ):
        super().__init__(
            targets=None,
            arg_value=arg_value,
            arg_label=arg_label,
            classical_controls=classical_controls,
            classical_control_value=classical_control_value,
            style=style
        )

    def get_compact_qobj(self):
        raise NotImplementedError(
            "GlobalPhase gate has no compack qobj representation."
        )

    def get_qobj(self, num_qubits):
        theta = self.arg_value
        N = 2**num_qubits

        return Qobj(
            np.exp(1.0j * theta) * sp.eye(N, N, dtype=complex, format="csr"),
            dims=[[2] * num_qubits, [2] * num_qubits],
        )


class TOFFOLI(ControlledGate):
    """
    TOFFOLI gate.

    Examples
    --------
    >>> from qutip_qip.operations import TOFFOLI
    >>> TOFFOLI([0, 1, 2]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 1. 0.]]
    """

    latex_str = r"{\rm TOFFOLI}"

    def __init__(
        self,
        targets,
        controls,
        control_value=None,
        classical_controls=None,
        classical_control_value=None,
        style=None,
    ):
        super().__init__(
            target_gate=X,
            targets=targets,
            controls=controls,
            control_value=control_value,
            classical_controls=classical_controls,
            classical_control_value=classical_control_value,
            style=style,
        )

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dims=[[2, 2, 2], [2, 2, 2]],
        )


class FREDKIN(ControlledGate):
    """
    FREDKIN gate.

    Examples
    --------
    >>> from qutip_qip.operations import FREDKIN
    >>> FREDKIN([0, 1, 2]).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]]
    """

    latex_str = r"{\rm FREDKIN}"

    def __init__(
        self,
        targets,
        controls,
        control_value=None,
        classical_controls=None,
        classical_control_value=None,
        style=None,
    ):
        super().__init__(
            target_gate=SWAP,
            targets=targets,
            controls=controls,
            control_value=control_value,
            classical_controls=classical_controls,
            classical_control_value=classical_control_value,
            style=style,
        )

    @staticmethod
    def get_compact_qobj():
        return Qobj(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dims=[[2, 2, 2], [2, 2, 2]],
        )
