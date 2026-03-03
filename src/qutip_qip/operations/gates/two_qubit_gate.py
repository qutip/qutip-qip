from typing import Final, Type
from functools import cache, lru_cache
import warnings

import numpy as np
from qutip import Qobj

from qutip_qip.operations import Gate, ControlledGate, AngleParametricGate
from qutip_qip.operations.gates import (
    X,
    Y,
    Z,
    H,
    S,
    T,
    RX,
    RY,
    RZ,
    QASMU,
    PHASE,
)


class _TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    __slots__ = ()
    num_qubits: Final[int] = 2


class _TwoQubitParametricGate(AngleParametricGate):
    """Abstract two-qubit Parametric Gate (non-controlled)."""

    __slots__ = ()
    num_qubits: Final[int] = 2


class _ControlledTwoQubitGate(ControlledGate):
    """Abstract two-qubit Controlled Gate (both parametric and non-parametric)."""

    __slots__ = ()
    num_qubits: Final[int] = 2
    num_ctrl_qubits: Final[int] = 1
    ctrl_value: Final[int] = 1


class SWAP(_TwoQubitGate):
    """
    SWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SWAP
    >>> SWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm SWAP}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )


class ISWAP(_TwoQubitGate):
    """
    iSWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import ISWAP
    >>> ISWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+1.j 0.+0.j]
     [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{i}{\rm SWAP}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return ISWAPdag


class ISWAPdag(_TwoQubitGate):
    """
    iSWAPdag gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import ISWAPdag
    >>> ISWAPdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]
     [0.+0.j 0.-1.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{i}{\rm SWAP^\dagger}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return ISWAP


class SQRTSWAP(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{SWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTSWAP
    >>> SQRTSWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"\sqrt{\rm SWAP}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    def inverse() -> Type[Gate]:
        return SQRTSWAPdag


class SQRTSWAPdag(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{SWAP}}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTSWAPdag
    >>> SQRTSWAPdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"\sqrt{\rm SWAP}^\dagger"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    def inverse() -> Type[Gate]:
        return SQRTSWAP


class SQRTISWAP(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{iSWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTISWAP
    >>> SQRTISWAP.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j      0.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.j      0.     +0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     +0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     +0.j      0.     +0.j      0.     +0.j      1.     +0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"\sqrt{{i}\rm SWAP}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    def inverse() -> Type[Gate]:
        return SQRTISWAPdag


class SQRTISWAPdag(_TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{iSWAP}}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTISWAPdag
    >>> SQRTISWAPdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j      0.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.j      0.     -0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     -0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     +0.j      0.     +0.j      0.     +0.j      1.     +0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"\sqrt{{i}\rm SWAP}^\dagger"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    def inverse() -> Type[Gate]:
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
    >>> from qutip_qip.operations.gates import BERKELEY
    >>> BERKELEY.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92388+0.j      0.     +0.j      0.     +0.j      0.     +0.38268j]
     [0.     +0.j      0.38268+0.j      0.     +0.92388j 0.     +0.j     ]
     [0.     +0.j      0.     +0.92388j 0.38268+0.j      0.     +0.j     ]
     [0.     +0.38268j 0.     +0.j      0.     +0.j      0.92388+0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm BERKELEY}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    def inverse() -> Type[Gate]:
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
    >>> from qutip_qip.operations.gates import BERKELEYdag
    >>> BERKELEYdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92388+0.j      0.     +0.j      0.     +0.j      0.     -0.38268j]
     [0.     +0.j      0.38268+0.j      0.     -0.92388j 0.     +0.j     ]
     [0.     +0.j      0.     -0.92388j 0.38268+0.j      0.     +0.j     ]
     [0.     -0.38268j 0.     +0.j      0.     +0.j      0.92388+0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm BERKELEY^\dagger}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    def inverse() -> Type[Gate]:
        return BERKELEY


class SWAPALPHA(_TwoQubitParametricGate):
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
    >>> from qutip_qip.operations.gates import SWAPALPHA
    >>> SWAPALPHA(0.5).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = "alpha"

    num_params: Final[int] = 1
    latex_str: Final[str] = r"{\rm SWAPALPHA}"

    def __init__(self, alpha: float, arg_label: str | None = None):
        super().__init__(alpha, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        alpha = arg_value[0]
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

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float]]:

        alpha = self.arg_value[0]
        if expanded:
            return SWAPALPHA, (-alpha,)
        return SWAPALPHA(-alpha)


class MS(_TwoQubitParametricGate):
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
    >>> from qutip_qip.operations.gates import MS
    >>> MS((np.pi/2, 0)).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     +0.j      0.     +0.j      0.     -0.70711j]
     [0.     +0.j      0.70711+0.j      0.     -0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     -0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     -0.70711j 0.     +0.j      0.     +0.j      0.70711+0.j     ]]
    """

    __slots__ = ("theta", "phi")

    num_params: Final[int] = 2
    latex_str: Final[str] = r"{\rm MS}"

    def __init__(self, theta: float, phi: float, arg_label: str | None = None):
        super().__init__(theta, phi, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float, float]) -> Qobj:
        theta, phi = arg_value
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

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float, float]]:

        theta, phi = self.arg_value
        inverse_param = (-theta, phi)

        if expanded:
            return MS, inverse_param
        return MS(*inverse_param)


class RZX(_TwoQubitParametricGate):
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
    >>> from qutip_qip.operations.gates import RZX
    >>> RZX(np.pi).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.+0.j 0.-1.j 0.+0.j 0.+0.j]
    [0.-1.j 0.+0.j 0.+0.j 0.+0.j]
    [0.+0.j 0.+0.j 0.+0.j 0.+1.j]
    [0.+0.j 0.+0.j 0.+1.j 0.+0.j]]
    """

    __slots__ = "theta"

    num_params: Final[int] = 1
    latex_str: Final[str] = r"{\rm RZX}"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        theta = arg_value[0]
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

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float]]:

        theta = self.arg_value[0]
        if expanded:
            return RZX, (-theta,)
        return RZX(-theta)


class CX(_ControlledTwoQubitGate):
    """
    CNOT gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CX
    >>> CX.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = X
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm CNOT}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dims=[[2, 2], [2, 2]],
        )


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
    >>> from qutip_qip.operations.gates import CY
    >>> CY.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0j  0.+0j  0.+0j  0.+0j]
     [ 0.+0j  1.+0j  0.+0j  0.+0j]
     [ 0.+0j  0.+0j  0.+0j  0.-1j]
     [ 0+0j.  0.+0j  0.+1j. 0.+0j]]
    """

    __slots__ = ()

    is_clifford: Final[bool] = True
    target_gate: Final[Type[Gate]] = Y
    latex_str: Final[str] = r"{\rm CY}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            dims=[[2, 2], [2, 2]],
        )


class CZ(_ControlledTwoQubitGate):
    """
    Controlled Z gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CZ
    >>> CZ.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = Z
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm CZ}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dims=[[2, 2], [2, 2]],
        )


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
    >>> from qutip_qip.operations.gates import CH
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = H
    latex_str: Final[str] = r"{\rm CH}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    >>> from qutip_qip.operations.gates import CT
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = T
    latex_str: Final[str] = r"{\rm CT}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
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
    >>> from qutip_qip.operations.gates import CS
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = S
    latex_str: Final[str] = r"{\rm CS}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj(
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]
            ),
            dims=[[2, 2], [2, 2]],
        )


class CRX(_ControlledTwoQubitGate):
    r"""
    Controlled X rotation.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CRX
    """

    __slots__ = ()

    num_params: Final[int] = 1
    target_gate: Final[Type[Gate]] = RX
    latex_str: Final[str] = r"{\rm CRX}"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dims=[[2, 2], [2, 2]],
        )

    def get_qobj(self) -> Qobj:
        return self._compute_qobj(self.arg_value)


class CRY(_ControlledTwoQubitGate):
    r"""
    Controlled Y rotation.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CRY
    """

    __slots__ = ()

    num_params: Final[int] = 1
    target_gate: Final[Type[Gate]] = RY
    latex_str: Final[str] = r"{\rm CRY}"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(theta / 2), -np.sin(theta / 2)],
                [0, 0, np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dims=[[2, 2], [2, 2]],
        )

    def get_qobj(self) -> Qobj:
        return self._compute_qobj(self.arg_value)


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
    >>> from qutip_qip.operations.gates import CRZ
    >>> CRZ(np.pi).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    target_gate: Final[Type[Gate]] = RZ
    latex_str: Final[str] = r"{\rm CRZ}"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * theta / 2), 0],
                [0, 0, 0, np.exp(1j * theta / 2)],
            ],
            dims=[[2, 2], [2, 2]],
        )

    def get_qobj(self) -> Qobj:
        return self._compute_qobj(self.arg_value)


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
    >>> from qutip_qip.operations.gates import CPHASE
    >>> CPHASE(np.pi/2).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    target_gate: Final[Type[Gate]] = PHASE
    latex_str: Final[str] = r"{\rm CPHASE}"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * theta)],
            ],
            dims=[[2, 2], [2, 2]],
        )

    def get_qobj(self) -> Qobj:
        return self._compute_qobj(self.arg_value)


class CQASMU(_ControlledTwoQubitGate):
    r"""
    Controlled QASMU rotation.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CQASMU
    """

    __slots__ = ()

    num_params: Final[int] = 3
    target_gate: Final[Type[Gate]] = QASMU
    latex_str: Final[str] = r"{\rm CQASMU}"

    def __init__(
        self,
        theta: float,
        phi: float,
        gamma: float,
        arg_label: str | None = None,
    ):
        super().__init__(theta, phi, gamma, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float, float, float]) -> Qobj:
        theta, phi, gamma = arg_value
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [
                    0,
                    0,
                    np.exp(-1j * (phi + gamma) / 2) * np.cos(theta / 2),
                    -np.exp(-1j * (phi - gamma) / 2) * np.sin(theta / 2),
                ],
                [
                    0,
                    0,
                    np.exp(1j * (phi - gamma) / 2) * np.sin(theta / 2),
                    np.exp(1j * (phi + gamma) / 2) * np.cos(theta / 2),
                ],
            ],
            dims=[[2, 2], [2, 2]],
        )

    def get_qobj(self) -> Qobj:
        return self._compute_qobj(self.arg_value)
