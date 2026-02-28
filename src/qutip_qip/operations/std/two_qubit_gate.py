from typing import Final
import warnings

import numpy as np
from qutip import Qobj

from qutip_qip.operations import Gate, ControlledGate, AngleParametricGate
from qutip_qip.operations.std import X, Y, Z, H, S, T, RX, RY, RZ, QASMU, PHASE


class _TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    __slots__ = ()
    num_qubits: Final[int] = 2


class _ControlledTwoQubitGate(ControlledGate):
    """
    This class allows correctly generating the gate instance
    when a redundant control_value is given, e.g.
    ``CX``,
    and raise an error if it is 0.
    """

    __slots__ = ()
    num_qubits: Final[int] = 2
    num_ctrl_qubits: Final[int] = 1
    ctrl_value: Final[int] = 1


class SWAP(_TwoQubitGate):
    """
    SWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SWAP
    >>> SWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]
    """

    __slots__ = ()

    self_inverse = True
    is_clifford = True
    latex_str = r"{\rm SWAP}"

    @staticmethod
    def get_qobj():
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )


class ISWAP(_TwoQubitGate):
    """
    iSWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import ISWAP
    >>> ISWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+1.j 0.+0.j]
     [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"{i}{\rm SWAP}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse_gate() -> Gate:
        return ISWAPdag


class ISWAPdag(_TwoQubitGate):
    """
    iSWAPdag gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import ISWAPdag
    >>> ISWAPdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]
     [0.+0.j 0.-1.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"{i}{\rm SWAP^\dagger}"

    @staticmethod
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse_gate() -> Gate:
        return ISWAP


class SQRTSWAP(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{SWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SQRTSWAP
    >>> SQRTSWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = ()

    self_inverse = False
    latex_str = r"\sqrt{\rm SWAP}"

    @staticmethod
    def get_qobj():
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

    @staticmethod
    def inverse_gate() -> Gate:
        return SQRTSWAPdag


class SQRTSWAPdag(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{SWAP}}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SQRTSWAPdag
    >>> SQRTSWAPdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = ()

    self_inverse = False
    latex_str = r"\sqrt{\rm SWAP}^\dagger"

    @staticmethod
    def get_qobj():
        return Qobj(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                    [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                    [0, 0, 0, 1],
                ]
            ),
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse_gate() -> Gate:
        return SQRTSWAP


class SQRTISWAP(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{iSWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SQRTISWAP
    >>> SQRTISWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j      0.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.j      0.     +0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     +0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     +0.j      0.     +0.j      0.     +0.j      1.     +0.j     ]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"\sqrt{{i}\rm SWAP}"

    @staticmethod
    def get_qobj():
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

    @staticmethod
    def inverse_gate() -> Gate:
        return SQRTISWAPdag


class SQRTISWAPdag(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{iSWAP}}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import SQRTISWAPdag
    >>> SQRTISWAPdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j      0.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.j      0.     -0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     -0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     +0.j      0.     +0.j      0.     +0.j      1.     +0.j     ]]
    """

    __slots__ = ()

    self_inverse = False
    is_clifford = True
    latex_str = r"\sqrt{{i}\rm SWAP}^\dagger"

    @staticmethod
    def get_qobj():
        return Qobj(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1 / np.sqrt(2), -1j / np.sqrt(2), 0],
                    [0, -1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 0, 0, 1],
                ]
            ),
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse_gate() -> Gate:
        return SQRTISWAP


class BERKELEY(_TwoQubitGate):
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
    >>> from qutip_qip.operations.std import BERKELEY
    >>> BERKELEY.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92388+0.j      0.     +0.j      0.     +0.j      0.     +0.38268j]
     [0.     +0.j      0.38268+0.j      0.     +0.92388j 0.     +0.j     ]
     [0.     +0.j      0.     +0.92388j 0.38268+0.j      0.     +0.j     ]
     [0.     +0.38268j 0.     +0.j      0.     +0.j      0.92388+0.j     ]]
    """

    __slots__ = ()

    self_inverse = False
    latex_str = r"{\rm BERKELEY}"

    @staticmethod
    def get_qobj():
        return Qobj(
            [
                [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
                [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
                [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
            ],
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse_gate() -> Gate:
        return BERKELEYdag


class BERKELEYdag(_TwoQubitGate):
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
    >>> from qutip_qip.operations.std import BERKELEYdag
    >>> BERKELEYdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92388+0.j      0.     +0.j      0.     +0.j      0.     -0.38268j]
     [0.     +0.j      0.38268+0.j      0.     -0.92388j 0.     +0.j     ]
     [0.     +0.j      0.     -0.92388j 0.38268+0.j      0.     +0.j     ]
     [0.     -0.38268j 0.     +0.j      0.     +0.j      0.92388+0.j     ]]
    """

    __slots__ = ()

    self_inverse = False
    latex_str = r"{\rm BERKELEY^\dagger}"

    @staticmethod
    def get_qobj():
        return Qobj(
            [
                [np.cos(np.pi / 8), 0, 0, -1.0j * np.sin(np.pi / 8)],
                [0, np.cos(3 * np.pi / 8), -1.0j * np.sin(3 * np.pi / 8), 0],
                [0, -1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                [-1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
            ],
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse_gate() -> Gate:
        return BERKELEY


class SWAPALPHA(AngleParametricGate):
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
    >>> from qutip_qip.operations.std import SWAPALPHA
    >>> SWAPALPHA(0.5).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = ()

    num_qubits = 2
    num_params: int = 1
    latex_str = r"{\rm SWAPALPHA}"

    def get_qobj(self):
        alpha = self.arg_value[0]
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

    def inverse_gate(self) -> Gate:
        alpha = self.arg_value[0]
        return SWAPALPHA(-alpha)


class MS(AngleParametricGate):
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
    >>> from qutip_qip.operations.std import MS
    >>> MS((np.pi/2, 0)).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     +0.j      0.     +0.j      0.     -0.70711j]
     [0.     +0.j      0.70711+0.j      0.     -0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     -0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     -0.70711j 0.     +0.j      0.     +0.j      0.70711+0.j     ]]
    """

    __slots__ = ()

    num_qubits = 2
    num_params: int = 2
    latex_str = r"{\rm MS}"

    def get_qobj(self):
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

    def inverse_gate(self) -> Gate:
        theta, phi = self.arg_value
        return MS([-theta, phi])


class RZX(AngleParametricGate):
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
    >>> from qutip_qip.operations.std import RZX
    >>> RZX(np.pi).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.+0.j 0.-1.j 0.+0.j 0.+0.j]
    [0.-1.j 0.+0.j 0.+0.j 0.+0.j]
    [0.+0.j 0.+0.j 0.+0.j 0.+1.j]
    [0.+0.j 0.+0.j 0.+1.j 0.+0.j]]
    """

    __slots__ = ()

    num_qubits = 2
    num_params = 1
    latex_str = r"{\rm RZX}"

    def get_qobj(self):
        theta = self.arg_value[0]
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

    def inverse_gate(self) -> Gate:
        theta = self.arg_value[0]
        return RZX(-theta)


class CX(_ControlledTwoQubitGate):
    """
    CNOT gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import CX
    >>> CX.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]
    """

    __slots__ = ()

    target_gate = X
    is_clifford = True
    latex_str = r"{\rm CNOT}"


class CNOT(CX):
    __slots__ = ()

    def __init__(self):
        warnings.warn(
            "CNOT is deprecated and will be removed in future versions. "
            "Use CX  instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()


class CY(_ControlledTwoQubitGate):
    """
    Controlled CY gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import CY
    >>> CY().get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0j  0.+0j  0.+0j  0.+0j]
     [ 0.+0j  1.+0j  0.+0j  0.+0j]
     [ 0.+0j  0.+0j  0.+0j  0.-1j]
     [ 0+0j.  0.+0j  0.+1j. 0.+0j]]
    """

    __slots__ = ()

    is_clifford = True
    target_gate = Y
    latex_str = r"{\rm CY}"


class CZ(_ControlledTwoQubitGate):
    """
    Controlled Z gate.

    Examples
    --------
    >>> from qutip_qip.operations.std import CZ
    >>> CZ().get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """

    __slots__ = ()

    target_gate = Z
    is_clifford = True
    latex_str = r"{\rm CZ}"


class CSIGN(CZ):
    __slots__ = ()

    def __init__(self):
        warnings.warn(
            "CSIGN is deprecated and will be removed in future versions. "
            "Use CZ instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()


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
    >>> from qutip_qip.operations.std import CH
    """

    __slots__ = ()

    target_gate = H
    latex_str = r"{\rm CH}"


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
    >>> from qutip_qip.operations.std import CT
    """

    __slots__ = ()

    target_gate = T
    latex_str = r"{\rm CT}"


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
    >>> from qutip_qip.operations.std import CS
    """

    __slots__ = ()

    target_gate = S
    latex_str = r"{\rm CS}"


class CPHASE(_ControlledTwoQubitGate):
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
    >>> from qutip_qip.operations.std import CPHASE
    >>> CPHASE(np.pi/2).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    __slots__ = ()

    num_params: int = 1
    target_gate = PHASE
    latex_str = r"{\rm CPHASE}"


class CRX(_ControlledTwoQubitGate):
    r"""
    Controlled X rotation.

    Examples
    --------
    >>> from qutip_qip.operations.std import CRX
    """

    __slots__ = ()

    num_params: int = 1
    target_gate = RX
    latex_str = r"{\rm CRX}"


class CRY(_ControlledTwoQubitGate):
    r"""
    Controlled Y rotation.

    Examples
    --------
    >>> from qutip_qip.operations.std import CRY
    """

    __slots__ = ()

    num_params: int = 1
    target_gate = RY
    latex_str = r"{\rm CRY}"


class CRZ(_ControlledTwoQubitGate):
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
    >>> from qutip_qip.operations.std import CRZ
    >>> CRZ(np.pi).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    __slots__ = ()

    num_params: int = 1
    target_gate = RZ
    latex_str = r"{\rm CRZ}"


class CQASMU(_ControlledTwoQubitGate):
    r"""
    Controlled QASMU rotation.

    Examples
    --------
    >>> from qutip_qip.operations.std import CQASMU
    """

    __slots__ = ()

    num_params: int = 3
    target_gate = QASMU
    latex_str = r"{\rm CQASMU}"
