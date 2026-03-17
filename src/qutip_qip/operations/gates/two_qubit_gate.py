from abc import abstractmethod
from functools import cache, lru_cache
from typing import Final, Type

import numpy as np
from qutip import Qobj

from qutip_qip.operations import Gate, ControlledGate, AngleParametricGate
from qutip_qip.operations.gates import (
    X,
    Y,
    Z,
    H,
    S,
    Sdag,
    T,
    Tdag,
    RX,
    RY,
    RZ,
    QASMU,
    PHASE,
)
from qutip_qip.operations.namespace import NS_GATE


class _TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    __slots__ = ()
    namespace = NS_GATE
    num_qubits: Final[int] = 2


class _TwoQubitParametricGate(AngleParametricGate):
    """Abstract two-qubit Parametric Gate (non-controlled)."""

    __slots__ = ()
    namespace = NS_GATE
    num_qubits: Final[int] = 2

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype)

    @staticmethod
    @abstractmethod
    def compute_qobj(args: tuple[float], dtype: str) -> Qobj:
        raise NotImplementedError


class _ControlledTwoQubitGate(ControlledGate):
    """Abstract two-qubit Controlled Gate (both parametric and non-parametric)."""

    __slots__ = ()
    namespace = NS_GATE

    num_qubits: Final[int] = 2
    num_ctrl_qubits: Final[int] = 1
    ctrl_value: Final[int] = 1


class SWAP(_TwoQubitGate):
    r"""
    SWAP gate.

    The SWAP gate exchanges the quantum states of two qubits.
    If the first qubit is in state $|\psi\rangle$ and the second qubit
    is in state $|\phi\rangle$, the application of this gate transforms the
    joint state $SWAP(|\psi\rangle \otimes |\phi\rangle)$ -> $|\phi\rangle \otimes |\psi\rangle$.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SWAP
    >>> SWAP.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )


class ISWAP(_TwoQubitGate):
    r"""
    Two-qubit iSWAP gate.

    The iSWAP gate exchanges the quantum states of two qubits while simultaneously
    applying a relative phase of $i$ (equivalent to a $\pi/2$ phase shift)
    to the $|01\rangle$ and $|10\rangle$ computational basis states.

    This gate is particularly used in hardware architectures dominated by
    XY-exchange interactions, such as superconducting qubits, where it
    serves as the native hardware-level entangling gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import ISWAP
    >>> ISWAP.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return ISWAPdag


class ISWAPdag(_TwoQubitGate):
    r"""
    Inverse iSWAP (iSWAP dagger) gate.

    This gate is the Hermitian conjugate, and therefore the inverse, of the
    standard iSWAP gate. It exchanges the quantum states of two distinct qubits
    while applying a relative phase of $-i$ (equivalent to a $-\pi/2$ phase shift)
    to the $|01\rangle$ and $|10\rangle$ computational basis states.

    Examples
    --------
    >>> from qutip_qip.operations.gates import ISWAPdag
    >>> ISWAPdag.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j 0.+0.j]
     [ 0.+0.j  0.+0.j -0.-1.j 0.+0.j]
     [ 0.+0.j -0.-1.j  0.+0.j 0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j 1.+0.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{i}{\rm SWAP^\dagger}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return ISWAP


class SQRTSWAP(_TwoQubitGate):
    r"""
    Two-qubit square root of SWAP (:math:`\sqrt{\mathrm{SWAP}}`) gate.

    This gate performs half of a standard SWAP operation. Logically, applying 
    this gate twice in succession exactly reproduces the behavior of a full 
    SWAP gate. It is a universal two-qubit entangling gate. 
    
    Physically, the :math:`\sqrt{\mathrm{SWAP}}` gate is used in 
    spin-based solid-state quantum computing architectures (such as quantum dots). 
    In these systems, pulsing the Heisenberg exchange interaction between two 
    neighboring spins for exactly half the duration required for a full SWAP 
    natively yields this operation.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & \frac{1+i}{2} & \frac{1-i}{2} & 0 \\
        0 & \frac{1-i}{2} & \frac{1+i}{2} & 0 \\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTSWAP
    >>> SQRTSWAP.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
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
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTSWAPdag


class SQRTSWAPdag(_TwoQubitGate):
    r"""
    Inverse (hermitian conjugate) of SQRTSWAP gate i.e. (:math:`\sqrt{\mathrm{SWAP}}^\dagger`).

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTSWAPdag
    >>> SQRTSWAPdag.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
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
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTSWAP


class SQRTISWAP(_TwoQubitGate):
    r"""
    Two-qubit square root of iSWAP (:math:`\sqrt{\mathrm{iSWAP}}`) gate.

    This gate performs half of a standard iSWAP operation. Logically, applying
    this gate twice in succession exactly reproduces the behavior of a full
    iSWAP gate. It is an universal entangling gate.

    Physically, the :math:`\sqrt{\mathrm{iSWAP}}` gate is a natural hardware-level
    operation for superconducting qubits coupled via capacitive or XY-exchange
    interactions.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTISWAP
    >>> SQRTISWAP.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.        +0.j         0.        +0.j         0.     +0.j ]
     [0.     +0.j      0.70710678+0.j         0.        +0.70710678j 0.     +0.j ]
     [0.     +0.j      0.        +0.70710678j 0.70710678+0.j         0.     +0.j ]
     [0.     +0.j      0.        +0.j         0.        +0.j         1.     +0.j ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"\sqrt{{i}\rm SWAP}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
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
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTISWAPdag


class SQRTISWAPdag(_TwoQubitGate):
    r"""
    Inverse (hermitian conjugate) of SQRTISWAP gate i.e. (:math:`\sqrt{\mathrm{ISWAP}}^\dagger`).

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTISWAPdag
    >>> SQRTISWAPdag.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.         +0.j           0.        +0.j           0.        +0.j         0.     +0.j     ]
     [ 0.         +0.j           0.70710678+0.j          -0.        -0.70710678j 0.     +0.j     ]
     [ 0.         +0.j          -0.        -0.70710678j   0.70710678+0.j         0.     +0.j     ]
     [ 0.         +0.j           0.        +0.j           0.        +0.j         1.     +0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"\sqrt{{i}\rm SWAP}^\dagger"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
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
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTISWAP


class BERKELEY(_TwoQubitGate):
    r"""
    Two-qubit Berkeley (B) gate.

    The Berkeley gate is a universal two-qubit entangling gate. 
    It is optimized for hardware architectures where the natural two-qubit 
    evolution is driven by the anisotropic XY exchange interaction. 

    Any arbitrary two-qubit unitary can be constructed using at 
    most two applications of the Berkeley gate interleaved with local single-qubit 
    rotations, making it theoretically more efficient than the standard CNOT gate 
    (which can require up to three applications).

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
    >>> BERKELEY.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92387953+0.j         0.        +0.j         0.        +0.j         0.        +0.38268343j]
     [0.        +0.j         0.38268343+0.j         0.        +0.92387953j 0.        +0.j     ]
     [0.        +0.j         0.        +0.92387953j 0.38268343+0.j         0.        +0.j     ]
     [0.        +0.38268343j 0.        +0.j         0.        +0.j         0.92387953+0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm BERKELEY}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
                [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
                [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return BERKELEYdag


class BERKELEYdag(_TwoQubitGate):
    r"""
    Inverse (hermitian conjugate) of BERKLEY gate.

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
    >>> BERKELEYdag.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92387953+0.j         0.        +0.j         0.        +0.j         0.        -0.38268343j]
     [0.        +0.j         0.38268343+0.j         0.        -0.92387953j 0.        +0.j     ]
     [0.        +0.j         0.        -0.92387953j 0.38268343+0.j         0.        +0.j     ]
     [0.        -0.38268343j 0.        +0.j         0.        +0.j         0.92387953+0.j     ]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm BERKELEY^\dagger}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [np.cos(np.pi / 8), 0, 0, -1.0j * np.sin(np.pi / 8)],
                [0, np.cos(3 * np.pi / 8), -1.0j * np.sin(3 * np.pi / 8), 0],
                [0, -1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                [-1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return BERKELEY


class SWAPALPHA(_TwoQubitParametricGate):
    r"""
    Two-qubit parameterized fractional SWAP (SWAP-alpha) gate.

    This gate applies a continuous exchange interaction between two qubits, 
    parameterized by a dimensionless variable $\alpha$. It represents a fractional 
    power of the standard SWAP operator. 

    When $\alpha = 1$, the gate acts as a standard SWAP gate. When $\alpha = 0.5$, 
    it acts precisely as the universal $\sqrt{\mathrm{SWAP}}$ gate. This continuous 
    parameterization is used in modeling physical systems governed by 
    the Heisenberg exchange interaction (such as coupled quantum dots), where the 
    interaction time dictates the fractional swapping of states.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & \frac{1 + e^{i\pi\alpha}}{2} & \frac{1 - e^{i\pi\alpha}}{2} & 0 \\
        0 & \frac{1 - e^{i\pi\alpha}}{2} & \frac{1 + e^{i\pi\alpha}}{2} & 0 \\
        0 & 0 & 0 & 1
        \end{pmatrix}

    Parameters
    ----------
    arg_value : float or tuple[float]
        The fractional power parameter $\alpha$.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SWAPALPHA
    >>> SWAPALPHA(0.5).get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

    __slots__ = ("alpha",)

    num_params: Final[int] = 1
    latex_str: Final[str] = r"{\rm SWAPALPHA}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
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
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float]]:
        alpha = self.arg_value[0]
        return SWAPALPHA((-alpha))


class MS(_TwoQubitParametricGate):
    r"""
    Two-qubit parameterized Mølmer–Sørensen (MS) gate.

    The Mølmer–Sørensen gate is the primary native entangling operation for 
    trapped-ion quantum computers.

    Mathematically, it generates a rotation by an angle :math:`\theta` 
    around an axis in the equatorial plane of the two-qubit Bloch sphere, 
    determined by the phase :math:`\phi` of the driving fields. When 
    :math:`\theta = \pi/2` and :math:`\phi = 0`, it produces a maximally 
    entangled state, functioning identically to an XX-gate.

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & 0 & 0 & -ie^{-i2\phi}\sin(\frac{\theta}{2}) \\
        0 & \cos(\frac{\theta}{2}) & -i\sin(\frac{\theta}{2}) & 0 \\
        0 & -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) & 0 \\
        -ie^{i2\phi}\sin(\frac{\theta}{2}) & 0 & 0 & \cos(\frac{\theta}{2})
        \end{pmatrix}

    Parameters
    ----------
    arg_value : tuple[float, float]
        A tuple of two floats `(theta, phi)`, representing the rotation angle 
        :math:`\theta` and the phase angle :math:`\phi` in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import MS
    >>> MS((np.pi/2, 0)).get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70710678+0.j         0.        +0.j         0.        +0.j         0.        -0.70710678j]
     [0.        +0.j         0.70710678+0.j         0.        -0.70710678j 0.        +0.j     ]
     [0.        +0.j         0.        -0.70710678j 0.70710678+0.j         0.        +0.j     ]
     [0.        -0.70710678j 0.        +0.j         0.        +0.j         0.70710678+0.j     ]]
    """

    __slots__ = ("theta", "phi")

    num_params: Final[int] = 2
    latex_str: Final[str] = r"{\rm MS}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float, float], dtype: str) -> Qobj:
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
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float, float]]:
        theta, phi = self.arg_value
        return MS((-theta, phi))


class RZX(_TwoQubitParametricGate):
    r"""
    Two-qubit parameterized RZX gate.

    This gate performs a continuous rotation by an angle :math:`\theta` generated 
    by the :math:`Z \otimes X` Pauli interaction. 

    Physically, this gate is used in the context of superconducting qubits.
    It is the direct unitary evolution generated by the cross-resonance 
    effect, where driving the control qubit at the frequency of the target qubit 
    creates an effective :math:`Z \otimes X` coupling. It serves as the fundamental 
    building block for constructing CNOT gates on such hardware architectures.

    .. math::

        \begin{pmatrix}
        \cos{\theta/2} & -i\sin{\theta/2} & 0 & 0 \\
        -i\sin{\theta/2} & \cos{\theta/2} & 0 & 0 \\
        0 & 0 & \cos{\theta/2} & i\sin{\theta/2} \\
        0 & 0 & i\sin{\theta/2} & \cos{\theta/2} \\
        \end{pmatrix}

    Parameters
    ----------
    arg_value : float or tuple[float]
        A float or a tuple containing a single float `theta`, representing the rotation 
        angle :math:`\theta` in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RZX
    >>> RZX(np.pi).get_qobj().tidyup()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.+0.j 0.-1.j 0.+0.j 0.+0.j]
    [0.-1.j 0.+0.j 0.+0.j 0.+0.j]
    [0.+0.j 0.+0.j 0.+0.j 0.+1.j]
    [0.+0.j 0.+0.j 0.+1.j 0.+0.j]]
    """

    __slots__ = ("theta",)

    num_params: Final[int] = 1
    latex_str: Final[str] = r"{\rm RZX}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
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
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float]]:
        theta = self.arg_value[0]
        return RZX((-theta))


class CX(_ControlledTwoQubitGate):
    """
    CNOT gate i.e. Controlled Pauli X gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CX
    >>> CX.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = X
    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm CNOT}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )


CNOT = CX


class CY(_ControlledTwoQubitGate):
    """
    Controlled Pauli Y gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CY
    >>> CY.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j   0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j   0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  -0.-1.j]
     [ 0.+0.j  0.+0.j  0.+1.j   0.+0.j]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = Y
    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm CY}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )


class CZ(_ControlledTwoQubitGate):
    """
    Controlled Pauli Z gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CZ
    >>> CZ.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = Z
    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm CZ}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )


CSIGN = CZ


class CH(_ControlledTwoQubitGate):
    r"""
    Controlled Hadamard gate.

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
    >>> CH.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.          0.          0.          0.        ]
     [ 0.          1.          0.          0.        ]
     [ 0.          0.          0.70710678  0.70710678]
     [ 0.          0.          0.70710678 -0.70710678]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = H
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm CH}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        sq_2 = 1 / np.sqrt(2)
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, sq_2, sq_2],
                [0, 0, sq_2, -sq_2],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )


class CT(_ControlledTwoQubitGate):
    r"""
    Controlled T gate.

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
    >>> CT.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j   0. +0.j ]
     [0. +0.j  1. +0.j  0. +0.j   0. +0.j ]
     [0. +0.j  0. +0.j  1. +0.j   0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j   0.70710678+0.70710678j]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = T
    latex_str: Final[str] = r"{\rm CT}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, (1 + 1j) / np.sqrt(2)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return CTdag


class CTdag(_ControlledTwoQubitGate):
    r"""
    Inverse of CT gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations.gates import CTdag
    >>> CTdag.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j   0. +0.j ]
     [0. +0.j  1. +0.j  0. +0.j   0. +0.j ]
     [0. +0.j  0. +0.j  1. +0.j   0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j   0.70710678-0.70710678j]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = Tdag
    latex_str: Final[str] = r"{\rm CT^\dagger}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, (1 - 1j) / np.sqrt(2)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return CT


class CS(_ControlledTwoQubitGate):
    r"""
    Controlled S gate.

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
    >>> CS.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j  0.+0.j  0.+0.j   0.+0.j]
     [0.+0.j  1.+0.j  0.+0.j   0.+0.j]
     [0.+0.j  0.+0.j  1.+0.j   0.+0.j]
     [0.+0.j  0.+0.j  0.+0.j   0.+1.j]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = S
    latex_str: Final[str] = r"{\rm CS}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]
            ),
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return CSdag


class CSdag(_ControlledTwoQubitGate):
    r"""
    Inverse of CS gate.

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Examples
    --------
    >>> from qutip_qip.operations.gates import CSdag
    >>> CSdag.get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j   0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j   0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j   0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j   -0.-1.j]]
    """

    __slots__ = ()

    target_gate: Final[Type[Gate]] = Sdag
    latex_str: Final[str] = r"{\rm CS^\dagger}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1j]]
            ),
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return CS


class CRX(_ControlledTwoQubitGate):
    r"""
    Two-qubit Controlled RX gate.

    Parameters
    ----------
    arg_value : float or tuple[float]
        A float or tuple containing a single float `theta`, representing the
        rotation angle :math:`\theta` in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CRX
    >>> from math import pi
    >>> CRX(pi).get_qobj().tidyup()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.-1.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    target_gate: Final[Type[Gate]] = RX
    latex_str: Final[str] = r"{\rm CRX}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype=dtype)

    def inverse(self) -> Gate:
        theta = self.arg_value[0]
        return CRX((-theta,))


class CRY(_ControlledTwoQubitGate):
    r"""
    Two-qubit Controlled RY gate.

    Parameters
    ----------
    arg_value : float or tuple[float]
        A float or tuple containing a single float `theta`, representing the
        rotation angle :math:`\theta` in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CRY
    >>> from math import pi
    >>> CRY(pi/2).get_qobj().tidyup()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.          0.          0.          0.        ]
     [ 0.          1.          0.          0.        ]
     [ 0.          0.          0.70710678 -0.70710678]
     [ 0.          0.          0.70710678  0.70710678]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    target_gate: Final[Type[Gate]] = RY
    latex_str: Final[str] = r"{\rm CRY}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(theta / 2), -np.sin(theta / 2)],
                [0, 0, np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype)

    def inverse(self) -> Gate:
        theta = self.arg_value[0]
        return CRY((-theta,))


class CRZ(_ControlledTwoQubitGate):
    r"""
    Two-qubit Controlled RZ gate.

    Parameters
    ----------
    arg_value : float or tuple[float]
        A float or tuple containing a single float `theta`, representing the
        rotation angle :math:`\theta` in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CRZ
    >>> from math import pi
    >>> CRZ(pi).get_qobj().tidyup()
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

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-1j * theta / 2), 0],
                [0, 0, 0, np.exp(1j * theta / 2)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype)

    def inverse(self) -> Gate:
        theta = self.arg_value[0]
        return CRZ((-theta,))


class CPHASE(_ControlledTwoQubitGate):
    r"""
    Two-qubit parameterized controlled phase (CPHASE) gate.

    This gate applies a phase shift of :math:`\theta` to the target qubit 
    if and only if the control qubit is in the :math:`|1\rangle` state.
    
    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & e^{i\theta} \\
        \end{pmatrix}

    Parameters
    ----------
    arg_value : float or tuple[float]
        A float or tuple containing a single float `theta`, representing the phase 
        shift angle :math:`\theta` in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CPHASE
    >>> from math import pi
    >>> CPHASE(pi/2).get_qobj().tidyup()
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

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        theta = arg_value[0]
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * theta)],
            ],
            dims=[[2, 2], [2, 2]],
            dtype=dtype,
        )

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype)

    def inverse(self) -> Gate:
        theta = self.arg_value[0]
        return CPHASE((-theta,))


class CQASMU(_ControlledTwoQubitGate):
    r"""
    Two-qubit controlled universal rotation (CQASMU) gate.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & e^{-i(\phi + \gamma)/2} \cos(\frac{\theta}{2}) & -e^{-i(\phi - \gamma)/2} \sin(\frac{\theta}{2}) \\
        0 & 0 & e^{i(\phi - \gamma)/2} \sin(\frac{\theta}{2}) & e^{i(\phi + \gamma)/2} \cos(\frac{\theta}{2})
        \end{pmatrix}

    Parameters
    ----------
    arg_value : tuple[float, float, float]
        A tuple of three floats `(theta, phi, gamma)` representing the Euler 
        angles in radians for the target rotation.

    Examples
    --------
    >>> from qutip_qip.operations.gates import CQASMU
    >>> from math import pi
    >>> CQASMU((pi/2, pi, pi/2)).get_qobj()
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1. +0.j   0. +0.j   0. +0.j   0. +0.j ]
     [ 0. +0.j   1. +0.j   0. +0.j   0. +0.j ]
     [ 0. +0.j   0. +0.j  -0.5-0.5j -0.5+0.5j]
     [ 0. +0.j   0. +0.j   0.5+0.5j -0.5+0.5j]]
    """

    __slots__ = ()

    num_params: Final[int] = 3
    target_gate: Final[Type[Gate]] = QASMU
    latex_str: Final[str] = r"{\rm CQASMU}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(
        arg_value: tuple[float, float, float], dtype: str
    ) -> Qobj:
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
            dtype=dtype,
        )

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype=dtype)

    def inverse(self) -> Gate:
        theta, phi, gamma = self.arg_value
        return CQASMU((-theta, -gamma, -phi))
