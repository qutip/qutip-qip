from functools import partial
import numpy as np

from qutip import Qobj, sigmax, sigmay, sigmaz
from qutip_qip.operations import (
    Gate,
    SingleQubitGate,
    TwoQubitGate,
    ControlledGate,
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"X"

    def get_compact_qobj(self):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"Y"

    def get_compact_qobj(self):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"Z"

    def get_compact_qobj(self):
        return sigmaz(dtype="dense")


class RX(SingleQubitGate):
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

    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"R_x"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj([
            [np.cos(phi / 2), -1j * np.sin(phi / 2)],
            [-1j * np.sin(phi / 2), np.cos(phi / 2)],
        ])


class RY(SingleQubitGate):
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

    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"R_y"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj([
            [np.cos(phi / 2), -np.sin(phi / 2)],
            [np.sin(phi / 2), np.cos(phi / 2)],
        ])


class RZ(SingleQubitGate):
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

    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"R_z"

    def get_compact_qobj(self):
        phi = self.arg_value
        return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])


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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm H}"

    def get_compact_qobj(self):
        return 1 / np.sqrt(2.0) * Qobj([[1, 1], [1, -1]])


SNOT = H
SNOT.__doc__ = H.__doc__


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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"\sqrt{\rm NOT}"

    def get_compact_qobj(self):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm S}"

    def get_compact_qobj(self):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm T}"

    def get_compact_qobj(self):
        return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])


class R(SingleQubitGate):
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

    def __init__(self, targets, arg_value=None, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"{\rm R}"

    def get_compact_qobj(self):
        phi, theta = self.arg_value
        return Qobj([
            [
                np.cos(theta / 2.0),
                -1.0j * np.exp(-1.0j * phi) * np.sin(theta / 2.0),
            ],
            [
                -1.0j * np.exp(1.0j * phi) * np.sin(theta / 2.0),
                np.cos(theta / 2.0),
            ],
        ])


class QASMU(SingleQubitGate):
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

    def __init__(self, targets, arg_value=None, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"{\rm QASM-U}"

    def get_compact_qobj(self):
        theta, phi, gamma = self.arg_value
        return Qobj([
            [
                np.exp(-1j*(phi+gamma)/2)*np.cos(theta/2),
                -np.exp(-1j*(phi-gamma)/2)*np.sin(theta/2)
            ],
            [
                np.exp(1j*(phi-gamma)/2)*np.sin(theta/2),
                np.exp(1j*(phi+gamma)/2)*np.cos(theta/2)
            ]
        ])


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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm SWAP}"

    def get_compact_qobj(self):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{i}{\rm SWAP}"

    def get_compact_qobj(self):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"\sqrt{\rm SWAP}"

    def get_compact_qobj(self):
        return Qobj(
            np.array([
                [1, 0, 0, 0],
                [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                [0, 0, 0, 1],
            ]),
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"\sqrt{{i}\rm SWAP}"

    def get_compact_qobj(self):
        return Qobj(
            np.array([
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]),
            dims=[[2, 2], [2, 2]]
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm BERKELEY}"

    def get_compact_qobj(self):
        return Qobj([
                [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
                [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
                [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
                [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
            ],
            dims=[[2, 2], [2, 2]],
        )


class SWAPALPHA(TwoQubitGate):
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

    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"{\rm SWAPALPHA}"

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


class MS(TwoQubitGate):
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

    def __init__(self, targets, arg_value, **kwargs):
        super().__init__(targets=targets, arg_value=arg_value, **kwargs)
        self.latex_str = r"{\rm MS}"

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


class RZX(TwoQubitGate):
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

    def __init__(self, targets, arg_value, **kwargs):
        self.target_gate = RZ
        super().__init__(
            targets=targets,
            arg_value=arg_value,
            target_gate=self.target_gate,
            **kwargs,
        )

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


class _OneControlledGate(ControlledGate, TwoQubitGate):
    """
    This class allows correctly generating the gate instance
    when a redundant control_value is given, e.g.
    ``CNOT(0, 1, control_value=1)``,
    and raise an error if it is 0.
    """

    def __init__(self, controls, targets, target_gate, **kwargs):
        _control_value = kwargs.get("control_value", None)
        if _control_value is not None:
            if _control_value != 1:
                raise ValueError(
                    f"{self.__class__.__name__} must has control_value=1"
                )
        else:
            kwargs["control_value"] = 1
        super().__init__(
            targets=targets,
            controls=controls,
            target_gate=target_gate,
            **kwargs,
        )


class CNOT(_OneControlledGate):
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

    def __init__(self, controls, targets, **kwargs):
        self.target_gate = X
        super().__init__(
            targets=targets,
            controls=controls,
            target_gate=self.target_gate,
            **kwargs,
        )
        self.latex_str = r"{\rm CNOT}"

    def get_compact_qobj(self):
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dims=[[2, 2], [2, 2]],
        )


class CZ(_OneControlledGate):
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

    def __init__(self, controls, targets, **kwargs):
        self.target_gate = Z
        super().__init__(
            targets=targets,
            controls=controls,
            target_gate=self.target_gate,
            **kwargs,
        )

    def get_compact_qobj(self):
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dims=[[2, 2], [2, 2]],
        )


class CSIGN(_OneControlledGate):
    """
    Controlled CSIGN gate. Identical to the CZ gate.

    Examples
    --------
    >>> from qutip_qip.operations import CSIGN
    >>> CSIGN(0, 1).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """

    def __init__(self, controls, targets, **kwargs):
        self.target_gate = Z
        super().__init__(
            targets=targets,
            controls=controls,
            target_gate=self.target_gate,
            **kwargs,
        )

    def get_compact_qobj(self):
        return Qobj(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dims=[[2, 2], [2, 2]],
        )


class CPHASE(_OneControlledGate):
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
    >>> CPHASE(0, 1, np.pi/2).get_compact_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    def __init__(
        self, controls, targets, arg_value, control_value=1, **kwargs
    ):
        self.target_gate = RZ
        super().__init__(
            targets=targets,
            controls=controls,
            arg_value=arg_value,
            target_gate=self.target_gate,
            **kwargs,
        )

    def get_compact_qobj(self):
        return Qobj([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j*self.arg_value)]
        ], dims=[[2, 2], [2, 2]])

CRY = partial(_OneControlledGate, target_gate=RY)
CRY.__doc__ = "Controlled Y rotation."
CRX = partial(_OneControlledGate, target_gate=RX)
CRX.__doc__ = "Controlled X rotation."
CRZ = partial(_OneControlledGate, target_gate=RZ)
CRZ.__doc__ = "Controlled Z rotation."
CY = partial(_OneControlledGate, target_gate=Y)
CY.__doc__ = "Controlled Y gate."
CX = partial(_OneControlledGate, target_gate=X)
CX.__doc__ = "Controlled X gate."
CT = partial(_OneControlledGate, target_gate=T)
CT.__doc__ = "Controlled T gate."
CS = partial(_OneControlledGate, target_gate=S)
CS.__doc__ = "Controlled S gate."

########################### Special Gates #########################

class GLOBALPHASE(Gate):
    """
    GLOBALPHASE gate.

    Examples
    --------
    >>> from qutip_qip.operations import GLOBALPHASE
    """

    def __init__(self, arg_value: float, arg_label: str = None):
        super().__init__(arg_value=arg_value, arg_label=arg_label)
        self.latex_str = r"{\rm GLOBALPHASE}"

    def get_compact_qobj(self):
        raise NotImplementedError(
            "GlobalPhase gate has no compack qobj representation."
        )


class TOFFOLI(Gate):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm TOFFOLI}"

    def get_compact_qobj(self):
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


class FREDKIN(Gate):
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

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        self.latex_str = r"{\rm FREDKIN}"

    def get_compact_qobj(self):
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
