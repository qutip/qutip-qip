import numpy as np

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye
from qutip_qip.operations import AngleParametricOp, Unitary


class _SingleQubitGate(Unitary):
    """Abstract one-qubit gate."""

    num_qreg: int = 1


class _SingleQubitParametricGate(AngleParametricOp):
    """Abstract one-qubit parametric gate."""

    num_qreg: int = 1


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

    self_inverse: bool = True
    is_clifford: bool = True
    latex_str: str = r"X"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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

    self_inverse: bool = True
    is_clifford: bool = True
    latex_str: str = r"Y"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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

    self_inverse: bool = True
    is_clifford: bool = True
    latex_str: str = r"Z"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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

    self_inverse: bool = True
    is_clifford: bool = True
    latex_str: str = r"{\rm I}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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

    self_inverse: bool = True
    is_clifford: bool = True
    latex_str: str = r"H"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
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

    self_inverse: bool = False
    is_clifford: bool = True
    latex_str: str = r"\sqrt{\rm X}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=dtype)

    @staticmethod
    def compute_inverse() -> type[Unitary]:
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

    self_inverse: bool = False
    is_clifford: bool = True
    latex_str: str = r"\sqrt{\rm X}^\dagger"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]], dtype=dtype)

    @staticmethod
    def compute_inverse() -> type[Unitary]:
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

    self_inverse: bool = False
    is_clifford: bool = True
    latex_str: str = r"{\rm S}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, 1j]], dtype=dtype)

    @staticmethod
    def compute_inverse() -> type[Unitary]:
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

    self_inverse: bool = False
    is_clifford: bool = True
    latex_str: str = r"{\rm S^\dagger}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, -1j]], dtype=dtype)

    @staticmethod
    def compute_inverse() -> type[Unitary]:
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

    self_inverse: bool = False
    latex_str: str = r"{\rm T}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=dtype)

    @staticmethod
    def compute_inverse() -> type[Unitary]:
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

    self_inverse: bool = False
    latex_str: str = r"{\rm Tdag}"

    @staticmethod
    def compute_qobj(dtype: str = "dense") -> Qobj:
        return Qobj([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=dtype)

    @staticmethod
    def compute_inverse() -> type[Unitary]:
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
    theta : float
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

    theta: float
    latex_str: str = r"R_x"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [np.cos(self.theta / 2), -1j * np.sin(self.theta / 2)],
                [-1j * np.sin(self.theta / 2), np.cos(self.theta / 2)],
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return RX(-self.theta)


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
    theta : float
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

    theta: float
    latex_str: str = r"R_y"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [np.cos(self.theta / 2), -np.sin(self.theta / 2)],
                [np.sin(self.theta / 2), np.cos(self.theta / 2)],
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return RY(-self.theta)


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
    theta : float
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

    theta: float
    latex_str: str = r"R_z"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return Qobj(
            [[np.exp(-1j * self.theta / 2), 0], [0, np.exp(1j * self.theta / 2)]],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return RZ(-self.theta)


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
    theta : float
        The phase shift angle (:math:`\theta`) in radians.

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

    theta: float
    latex_str: str = r"PHASE"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return Qobj(
            [[1, 0], [0, np.exp(1j * self.theta)]],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return PHASE(-self.theta)


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
    phi: float
        The phase angle

    theta : float
        The rotation angle
        
    Examples
    --------
    >>> from math import pi
    >>> from qutip_qip.operations.gates import R
    >>> R(pi/2, pi/2).get_qobj().tidyup()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]
    """

    phi: float
    theta: float
    latex_str: str = r"{\rm R}"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [
                    np.cos(self.theta / 2.0),
                    -1.0j * np.exp(-1.0j * self.phi) * np.sin(self.theta / 2.0),
                ],
                [
                    -1.0j * np.exp(1.0j * self.phi) * np.sin(self.theta / 2.0),
                    np.cos(self.theta / 2.0),
                ],
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return R(self.phi, -self.theta)


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
    >>> QASMU(pi/2, pi, pi/2).get_qobj()
    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[-0.5-0.5j -0.5+0.5j]
     [ 0.5+0.5j -0.5+0.5j]]
    """

    theta: float
    phi: float
    gamma: float
    latex_str: str = r"{\rm QASMU}"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return Qobj(
            [
                [
                    np.exp(-1j * (self.phi + self.gamma) / 2) * np.cos(self.theta / 2),
                    -np.exp(-1j * (self.phi - self.gamma) / 2) * np.sin(self.theta / 2),
                ],
                [
                    np.exp(1j * (self.phi - self.gamma) / 2) * np.sin(self.theta / 2),
                    np.exp(1j * (self.phi + self.gamma) / 2) * np.cos(self.theta / 2),
                ],
            ],
            dims=[[2], [2]],
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return QASMU(-self.theta, -self.gamma, -self.phi)


class IDLE(_SingleQubitParametricGate):
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

    T: float
    self_inverse: bool = True

    def validate_params(self):
        super().validate_params()
        if self.T < 0:
            raise ValueError(f"IDLE time must be non-negative, got {self.T}")

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        # Practically not required as this gate is only useful in pulse level
        # simulation, and the pulse compiler implementation of it will be
        # independent of get_qobj()
        return qeye(2, dtype=dtype)
