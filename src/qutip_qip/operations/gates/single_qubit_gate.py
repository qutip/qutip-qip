from abc import abstractmethod
from functools import cache, lru_cache
from typing import Final, Type
import numpy as np

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
from qutip_qip.operations import Gate, AngleParametricGate
from qutip_qip.operations.namespace import NS_GATE, NameSpace
from qutip_qip.typing import Real


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

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(self.arg_value, dtype)

    @staticmethod
    @abstractmethod
    def compute_qobj(args: tuple[float], dtype: str) -> Qobj:
        raise NotImplementedError


class X(_SingleQubitGate):
    """
    Pauli X gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import X
    >>> X.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return sigmax(dtype=dtype)


class Y(_SingleQubitGate):
    """
    Pauli Y gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import Y
    >>> Y.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return sigmay(dtype=dtype)


class Z(_SingleQubitGate):
    """
    Pauli Z gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import Z
    >>> Z.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return sigmaz(dtype=dtype)


class IDENTITY(_SingleQubitGate):
    """
    Single-Qubit IDENTITY gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import IDENTITY
    >>> IDENTITY.get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0.]
     [0. 1.]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm I}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return qeye(2, dtype=dtype)


class H(_SingleQubitGate):
    """
    Hadamard gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import H
    >>> H.get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70710678  0.70710678]
     [ 0.70710678 -0.70710678]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = True
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"H"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        sq_half = 1 / np.sqrt(2.0)
        return Qobj([[sq_half, sq_half], [sq_half, -sq_half]], dtype=dtype)


SNOT = H


class SQRTX(_SingleQubitGate):
    r"""
    :math:`\sqrt{X}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTX
    >>> SQRTX.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=dtype
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTXdag


class SQRTXdag(_SingleQubitGate):
    r"""
    :math:`\sqrt{X}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import SQRTXdag
    >>> SQRTXdag.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj(
            [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]], dtype=dtype
        )

    @staticmethod
    def inverse() -> Type[Gate]:
        return SQRTX


SQRTNOT = SQRTX


class S(_SingleQubitGate):
    r"""
    S gate i.e. :math:`\sqrt{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import S
    >>> S.get_qobj()
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
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, 1j]], dtype=dtype)

    @staticmethod
    def inverse() -> Type[Gate]:
        return Sdag


class Sdag(_SingleQubitGate):
    r"""
    Sdag gate i.e. :math:`(\sqrt{Z})^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import S
    >>> Sdag.get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.+0.j  0.+0.j]
     [ 0.+0.j -0.-1.j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    is_clifford: Final[bool] = True
    latex_str: Final[str] = r"{\rm S^\dagger}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, -1j]], dtype=dtype)

    @staticmethod
    def inverse() -> Type[Gate]:
        return S


class T(_SingleQubitGate):
    r"""
    T gate i.e. :math:`\sqrt[4]{Z}` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import T
    >>> T.get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70710678+0.70710678j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm T}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=dtype)

    @staticmethod
    def inverse() -> Type[Gate]:
        return Tdag


class Tdag(_SingleQubitGate):
    r"""
    Tdag gate i.e. :math:`\sqrt[4]{Z}^\dagger` gate.

    Examples
    --------
    >>> from qutip_qip.operations.gates import Tdag
    >>> Tdag.get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70710678-0.70710678j]]
    """

    __slots__ = ()

    self_inverse: Final[bool] = False
    latex_str: Final[str] = r"{\rm Tdag}"

    @staticmethod
    @cache
    def get_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=dtype)

    @staticmethod
    def inverse() -> Type[Gate]:
        return T


class RX(_SingleQubitParametricGate):
    r"""
    Single-qubit rotation gate about the X-axis.

    This parametric gate performs a rotation of the quantum state around the 
    X-axis of the Bloch sphere by a specified angle.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & -i\sin(\frac{\theta}{2}) \\
        -i\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}
    
    Parameters
    ----------
    arg_value : float or tuple[float]
        The rotation angle (theta) in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RX
    >>> from math import pi
    >>> RX(pi/2).get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70710678+0.j         0.        -0.70710678j]
     [0.        -0.70710678j 0.70710678+0.j        ]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    latex_str: Final[str] = r"R_x"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -1j * np.sin(phi / 2)],
                [-1j * np.sin(phi / 2), np.cos(phi / 2)],
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float]]:
        theta = self.arg_value[0]
        return RX((-theta,))


class RY(_SingleQubitParametricGate):
    r"""
    Single-qubit rotation gate about the Y-axis.

    This parametric gate performs a rotation of the quantum state around the 
    Y-axis of the Bloch sphere by a specified angle.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\
        \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}

    Parameters
    ----------
    arg_value : float or tuple[float]
        The rotation angle (theta) in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RY
    >>> from math import pi
    >>> RY(pi/2).get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    latex_str: Final[str] = r"R_y"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [
                [np.cos(phi / 2), -np.sin(phi / 2)],
                [np.sin(phi / 2), np.cos(phi / 2)],
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float]]:
        theta = self.arg_value[0]
        return RY((-theta,))


class RZ(_SingleQubitParametricGate):
    r"""
    Single-qubit rotation gate about the Z-axis.

    This parametric gate performs a rotation of the quantum state around the 
    Z-axis of the Bloch sphere by a specified angle.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        e^{-i\frac{\theta}{2}} & 0 \\
        0 & e^{i\frac{\theta}{2}}
        \end{pmatrix}

    Parameters
    ----------
    arg_value : float or tuple[float]
        The rotation angle (theta) in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import RZ
    >>> from math import pi
    >>> RZ(pi/2).get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70710678-0.70710678j 0.        +0.j        ]
     [0.        +0.j         0.70710678+0.70710678j]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    latex_str: Final[str] = r"R_z"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float]]:
        theta = self.arg_value[0]
        return RZ((-theta,))


class PHASE(_SingleQubitParametricGate):
    r"""
    Single-qubit parametric phase shift gate.

    This gate applies a phase shift of :math:`\phi` to the :math:`|1\rangle` state, 
    leaving the :math:`|0\rangle` state unchanged.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        1 & 0 \\
        0 & e^{i\phi}
        \end{pmatrix}

    Parameters
    ----------
    arg_value : float or tuple[float]
        The phase shift angle (:math:`\phi`) in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import PHASE
    >>> from math import pi
    >>> PHASE(pi/4).get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.        +0.j         0.        +0.j        ]
     [0.        +0.j         0.70710678+0.70710678j]]
    """

    __slots__ = ()

    num_params: Final[int] = 1
    latex_str: Final[str] = r"PHASE"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
        phi = arg_value[0]
        return Qobj(
            [[1, 0], [0, np.exp(1j * phi)]],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float]]:
        theta = self.arg_value[0]
        return PHASE((-theta,))


class R(_SingleQubitParametricGate):
    r"""
    Arbitrary single-qubit rotation gate.

    This gate performs a rotation by an angle :math:`\theta` about an axis in the 
    equatorial plane of the Bloch sphere, specified by the azimuthal angle :math:`\phi`.

    The matrix representation of this gate is:

    .. math::

        \begin{pmatrix}
        \cos(\frac{\theta}{2}) & -ie^{-i\phi} \sin(\frac{\theta}{2}) \\
        -ie^{i\phi} \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
        \end{pmatrix}

    Parameters
    ----------
    arg_value : tuple[float, float]
        A tuple of two floats `(phi, theta)`, representing the phase angle 
        :math:`\phi` and the rotation angle :math:`\theta` in radians.
        
    Examples
    --------
    >>> from math import pi
    >>> from qutip_qip.operations.gates import R
    >>> R(arg_value=(pi/2, pi/2)).get_qobj().tidyup()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]
    """

    __slots__ = ()

    num_params: Final[int] = 2
    latex_str: Final[str] = r"{\rm R}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float, float], dtype: str) -> Qobj:
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
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float, float]]:
        phi, theta = self.arg_value
        return R((phi, -theta))


class QASMU(_SingleQubitParametricGate):
    r"""
    Universal single-qubit rotation gate (OpenQASM standard).

    This gate implements the standard 3-parameter rotation defined in OpenQASM, 
    decomposed as a sequence of Z, Y, and Z rotations:
    
    .. math::
        U(\theta, \phi, \gamma) = RZ(\phi) RY(\theta) RZ(\gamma)

    The exact matrix representation computed by this gate is:

    .. math::

        \begin{pmatrix}
        e^{-i(\phi + \gamma)/2} \cos(\frac{\theta}{2}) & -e^{-i(\phi - \gamma)/2} \sin(\frac{\theta}{2}) \\
        e^{i(\phi - \gamma)/2} \sin(\frac{\theta}{2}) & e^{i(\phi + \gamma)/2} \cos(\frac{\theta}{2})
        \end{pmatrix}

    Parameters
    ----------
    arg_value : tuple[float, float, float]
        A tuple of three floats `(theta, phi, gamma)` representing the Euler 
        angles in radians.

    Examples
    --------
    >>> from qutip_qip.operations.gates import QASMU
    >>> from math import pi
    >>> QASMU(arg_value=(pi/2, pi, pi/2)).get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[-0.5-0.5j -0.5+0.5j]
     [ 0.5+0.5j -0.5+0.5j]]
    """

    __slots__ = ()

    num_params: Final[int] = 3
    latex_str: Final[str] = r"{\rm QASMU}"

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_qobj(arg_value: tuple[float], dtype: str) -> Qobj:
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
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def inverse(self) -> Gate | tuple[Type[Gate], tuple[float, float, float]]:
        theta, phi, gamma = self.arg_value
        return QASMU((-theta, -gamma, -phi))


class IDLE(AngleParametricGate):
    r"""
    Single-qubit idle operation (delay).

    This gate logically acts as the identity operator. However, in the context 
    of pulse-level simulation and compiler execution, it explicitly represents 
    a waiting period or delay of a specific duration on the target qubit.

    The matrix representation of this gate is the identity matrix:

    .. math::

        \begin{pmatrix}
        1 & 0 \\
        0 & 1
        \end{pmatrix}

    Parameters
    ----------
    T : float
        The non-negative time duration for the idle gate.
    arg_label : str, optional
        An optional string label for the gate argument

    Examples
    --------
    >>> from qutip_qip.operations.gates import IDLE
    >>> gate = IDLE(T=2.5)
    >>> gate.get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0.]
     [0. 1.]]
    """

    __slots__ = ()
    num_qubits = 1
    num_params = 1

    def __init__(self, T: float, arg_label=None):
        super().__init__(T, arg_label=arg_label)

    @staticmethod
    def validate_params(args):
        if not isinstance(args[0], Real):
            raise TypeError(f"{args[0]} must be a float")
        if args[0] < 0:
            raise ValueError(f"IDLE time must be non-negative, got {args[0]}")

    def get_qobj(self, dtype: str = "dense") -> Qobj:
        # Practically not required as this gate is only useful in pulse level
        # simulation, and the pulse compiler implementation of it will be
        # independent of get_qobj()
        return qeye(2, dtype=dtype)
