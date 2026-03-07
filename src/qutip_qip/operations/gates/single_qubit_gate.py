from functools import cache, lru_cache
from typing import Final, Type
import warnings
import numpy as np

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
from qutip_qip.operations import Gate, AngleParametricGate
from qutip_qip.operations.namespace import NS_GATE, NameSpace


class _SingleQubitGate(Gate):
    """Abstract one-qubit gate."""

    __slots__ = ()
    namespace: NameSpace = NS_GATE
    num_qubits: Final[int] = 1


class _SingleQubitParametricGate(AngleParametricGate):
    """Abstract one-qubit parametric gate."""

    __slots__ = ()
    namespace: NameSpace = NS_GATE
    num_qubits: Final[int] = 1


class X(_SingleQubitGate):
    """
    Single-qubit X gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import X
    >>> X.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0. 1.]
     [1. 0.]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"X"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return sigmax(dtype="dense")


class Y(_SingleQubitGate):
    """
    Single-qubit Y gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import Y
    >>> Y.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[0.+0.j 0.-1.j]
     [0.+1.j 0.+0.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"Y"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return sigmay(dtype="dense")


class Z(_SingleQubitGate):
    """
    Single-qubit Z gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import Z
    >>> Z.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.]
     [ 0. -1.]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"Z"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return sigmaz(dtype="dense")


class IDLE(_SingleQubitGate):
    """
    IDLE gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import IDLE
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm IDLE}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return qeye(2)


class H(_SingleQubitGate):
    """
    Hadamard gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import H
    >>> H.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70711  0.70711]
     [ 0.70711 -0.70711]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"H"

    @staticmethod
    @cache
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
    >>> from qutip_qip.operations.gates import SQRTX
    >>> SQRTX.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.5+0.5j 0.5-0.5j]
     [0.5-0.5j 0.5+0.5j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"\sqrt{\rm X}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTXdag


class SQRTXdag(_SingleQubitGate):
    r"""
    :math:`\sqrt{X}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTXdag
    >>> SQRTXdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.5-0.5j 0.5+0.5j]
     [0.5+0.5j 0.5-0.5j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"\sqrt{\rm X}^\dagger"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]])

    @staticmethod
    def inverse() -> Type[Gate]:
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
    >>> from qutip_qip.operations.gates import S
    >>> S.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.+1.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm S}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, 1j]])

    @staticmethod
    def inverse() -> Type[Gate]:
        return Sdag


class Sdag(_SingleQubitGate):
    r"""
    S gate or :math:`\sqrt{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import S
    >>> Sdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j]
     [0.+0.j 0.-1.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm S^\dagger}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, -1j]])

    @staticmethod
    def inverse() -> Type[Gate]:
        return S


class T(_SingleQubitGate):
    r"""
    T gate or :math:`\sqrt[4]{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import T
    >>> T.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm T}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    @staticmethod
    def inverse() -> Type[Gate]:
        return Tdag


class Tdag(_SingleQubitGate):
    r"""
    Tdag gate or :math:`\sqrt[4]{Z}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import Tdag
    >>> Tdag.get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711-0.70711j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm Tdag}"

    @staticmethod
    @cache
    def get_qobj() -> Qobj:
        return Qobj([[1, 0], [0, np.exp(-1j * np.pi / 4)]])

    @staticmethod
    def inverse() -> Type[Gate]:
        return T


class RX(_SingleQubitParametricGate):
    """
    Single-qubit rotation RX.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RX
    >>> RX(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     -0.70711j]
     [0.     -0.70711j 0.70711+0.j     ]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"R_x"

    def __init__(self, theta: float, arg_label=None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -1j * np.sin(phi / 2)],
                [-1j * np.sin(phi / 2), np.cos(phi / 2)],
            ],
            dims=[[2], [2]],
        )

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float]]:

        theta = self.arg_value[0]
        if expanded:
            return RX, (-theta,)
        return RX(-theta)


class RY(_SingleQubitParametricGate):
    """
    Single-qubit rotation RY.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RY
    >>> RY(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"R_y"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -np.sin(phi / 2)],
                [np.sin(phi / 2), np.cos(phi / 2)],
            ]
        )

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float]]:

        theta = self.arg_value[0]
        if expanded:
            return RY, (-theta,)
        return RY(-theta)


class RZ(_SingleQubitParametricGate):
    """
    Single-qubit rotation RZ.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RZ
    >>> RZ(3.14159/2).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711-0.70711j 0.     +0.j     ]
     [0.     +0.j      0.70711+0.70711j]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"R_z"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        phi = arg_value[0]
        return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float]]:

        theta = self.arg_value[0]
        if expanded:
            return RZ, (-theta,)
        return RZ(-theta)


class PHASE(_SingleQubitParametricGate):
    """
    PHASE Gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import PHASE
    """

    __slots__ = ()

    num_params: Final[int] = 1
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"PHASE"

    def __init__(self, theta: float, arg_label: str | None = None):
        super().__init__(theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [
                [1, 0],
                [0, np.exp(1j * phi)],
            ]
        )

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float]]:

        theta = self.arg_value[0]
        if expanded:
            return PHASE, (-theta,)
        return PHASE(-theta)


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
    >>> from qutip_qip.operations.gates import R
    >>> R(np.pi/2, np.pi/2).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70711 -0.70711]
     [ 0.70711  0.70711]]
    """

    __slots__ = ()

    num_params: Final[int] = 2
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm R}"

    def __init__(self, phi: float, theta: float, arg_label: str | None = None):
        super().__init__(phi, theta, arg_label=arg_label)

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_qobj(arg_value: tuple[float, float]) -> Qobj:
        phi, theta = arg_value
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

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float, float]]:

        phi, theta = self.arg_value
        inverse_params = (phi, -theta)

        if expanded:
            return R, inverse_params
        return R(*inverse_params)


class QASMU(_SingleQubitParametricGate):
    r"""
    QASMU gate.

    .. math::
        U(\theta, \phi, \gamma) = RZ(\phi) RY(\theta) RZ(\gamma)

    Examples
    --------
    >>> from qutip_qip.operations.gates import QASMU
    >>> QASMU(0, (np.pi/2, np.pi, np.pi/2)).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[-0.5-0.5j -0.5+0.5j]
     [ 0.5+0.5j -0.5+0.5j]]
    """

    __slots__ = ()

    num_params: Final[int] = 3
    self_inverse: Final[bool] = True
    latex_str: Final[str] = r"{\rm QASMU}"

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
    def _compute_qobj(arg_value: tuple[float]) -> Qobj:
        theta, phi, gamma = arg_value
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

    def inverse(
        self, expanded: bool = False
    ) -> Gate | tuple[Type[Gate], tuple[float, float, float]]:

        theta, phi, gamma = self.arg_value
        inverse_param = (-theta, -gamma, -phi)

        if expanded:
            return QASMU, inverse_param
        return QASMU(*inverse_param)
