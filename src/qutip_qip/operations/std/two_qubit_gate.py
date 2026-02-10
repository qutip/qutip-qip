from typing import Final

import numpy as np
import scipy.sparse as sp
from qutip import Qobj

from qutip_qip.operations import (
    TwoQubitGate,
    ControlledGate,
    ControlledParamGate,
)

class SWAP(TwoQubitGate):
    """
    SWAP gate.

    Examples
    --------
    >>> from qutip_qip.operations import SWAP
    >>> SWAP([0, 1]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]
    """

    latex_str = r"{\rm SWAP}"

    @staticmethod
    def get_qobj():
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
    >>> ISWAP([0, 1]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+1.j 0.+0.j]
     [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
    """

    latex_str = r"{i}{\rm SWAP}"

    @staticmethod
    def get_qobj():
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
    >>> SQRTSWAP([0, 1]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

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


class SQRTISWAP(TwoQubitGate):
    r"""
    :math:`\sqrt{\mathrm{iSWAP}}` gate.

    Examples
    --------
    >>> from qutip_qip.operations import SQRTISWAP
    >>> SQRTISWAP([0, 1]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.     +0.j      0.     +0.j      0.     +0.j      0.     +0.j     ]
     [0.     +0.j      0.70711+0.j      0.     +0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     +0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     +0.j      0.     +0.j      0.     +0.j      1.     +0.j     ]]
    """

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
    >>> BERKELEY([0, 1]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.92388+0.j      0.     +0.j      0.     +0.j      0.     +0.38268j]
     [0.     +0.j      0.38268+0.j      0.     +0.92388j 0.     +0.j     ]
     [0.     +0.j      0.     +0.92388j 0.38268+0.j      0.     +0.j     ]
     [0.     +0.38268j 0.     +0.j      0.     +0.j      0.92388+0.j     ]]
    """

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


class SWAPALPHA(_AngleParametricGate):
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
    >>> SWAPALPHA([0, 1], 0.5).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1. +0.j  0. +0.j  0. +0.j  0. +0.j ]
     [0. +0.j  0.5+0.5j 0.5-0.5j 0. +0.j ]
     [0. +0.j  0.5-0.5j 0.5+0.5j 0. +0.j ]
     [0. +0.j  0. +0.j  0. +0.j  1. +0.j ]]
    """

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


class MS(_AngleParametricGate):
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
    >>> MS([0, 1], (np.pi/2, 0)).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.70711+0.j      0.     +0.j      0.     +0.j      0.     -0.70711j]
     [0.     +0.j      0.70711+0.j      0.     -0.70711j 0.     +0.j     ]
     [0.     +0.j      0.     -0.70711j 0.70711+0.j      0.     +0.j     ]
     [0.     -0.70711j 0.     +0.j      0.     +0.j      0.70711+0.j     ]]
    """

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


class RZX(_AngleParametricGate):
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
    >>> RZX(np.pi).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[0.+0.j 0.-1.j 0.+0.j 0.+0.j]
    [0.-1.j 0.+0.j 0.+0.j 0.+0.j]
    [0.+0.j 0.+0.j 0.+0.j 0.+1.j]
    [0.+0.j 0.+0.j 0.+1.j 0.+0.j]]
    """

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


class _ControlledTwoQubitGate(ControlledGate):
    """
    This class allows correctly generating the gate instance
    when a redundant control_value is given, e.g.
    ``CNOT(0, 1, control_value=1)``,
    and raise an error if it is 0.
    """

    num_qubits: Final[int] = 2
    num_ctrl_qubits: Final[int] = 1


class CNOT(_ControlledTwoQubitGate):
    """
    CNOT gate.

    Examples
    --------
    >>> from qutip_qip.operations import CNOT
    >>> CNOT(0, 1).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]
    """

    target_gate = X
    latex_str = r"{\rm CNOT}"

    @staticmethod
    def get_qobj():
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
    >>> CSIGN(0, 1).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0j  0.+0j  0.+0j  0.+0j]
     [ 0.+0j  1.+0j  0.+0j  0.+0j]
     [ 0.+0j  0.+0j  0.+0j  0.-1j]
     [ 0+0j.  0.+0j  0.+1j. 0.+0j]]
    """

    latex_str = r"{\rm CY}"
    target_gate = Y

    @staticmethod
    def get_qobj():
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
    >>> CSIGN(0, 1).get_qobj() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0. -1.]]
    """

    latex_str = r"{\rm CZ}"
    target_gate = Z

    @staticmethod
    def get_qobj():
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
    target_gate = H

    @staticmethod
    def get_qobj():
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
    target_gate = T

    @staticmethod
    def get_qobj():
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
    target_gate = S

    @staticmethod
    def get_qobj():
        return Qobj(
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]]
            ),
            dims=[[2, 2], [2, 2]],
        )


class _ControlledParamTwoQubitGate(ControlledParamGate, _AngleParametricGate):
    """
    This class allows correctly generating the gate instance
    when a redundant control_value is given, e.g.
    ``CNOT(0, 1, control_value=1)``,
    and raise an error if it is 0.
    """

    num_qubits: Final[int] = 2
    num_ctrl_qubits: Final[int] = 1


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
    >>> CPHASE(0, 1, np.pi/2).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    num_params: int = 1
    target_gate = PHASE
    latex_str = r"{\rm CPHASE}"

    def get_qobj(self):
        return Qobj(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * self.arg_value[0])],
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

    num_params: int = 1
    target_gate = RX
    latex_str = r"{\rm CRX}"


class CRY(_ControlledParamTwoQubitGate):
    r"""
    Controlled Y rotation.

    Examples
    --------
    >>> from qutip_qip.operations import CRY
    """

    latex_str = r"{\rm CRY}"
    target_gate = RY
    num_params: int = 1


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
    >>> CRZ(0, 1, np.pi).get_qobj().tidyup() # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.-1.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]
    """

    num_params: int = 1
    target_gate = RZ
    latex_str = r"{\rm CRZ}"


class CQASMU(_ControlledParamTwoQubitGate):
    r"""
    Controlled QASMU rotation.

    Examples
    --------
    >>> from qutip_qip.operations import CQASMU
    """

    num_params: int = 3
    target_gate = QASMU
    latex_str = r"{\rm CQASMU}"


########################### Special Gates #########################


class GLOBALPHASE(_AngleParametricGate):
    """
    GLOBALPHASE gate.

    Examples
    --------
    >>> from qutip_qip.operations import GLOBALPHASE
    """

    num_qubits: int = 0
    num_params: int = 1
    latex_str = r"{\rm GLOBALPHASE}"

    def __init__(self, arg_value: float = 0.0):
        super().__init__(arg_value=arg_value)

    def __repr__(self):
        return f"Gate({self.name}, phase {self.arg_value})"

    def get_qobj(self, num_qubits=None):
        if num_qubits is None:
            return Qobj(self.arg_value)

        N = 2**num_qubits
        return Qobj(
            np.exp(1.0j * self.arg_value[0]) * sp.eye(N, N, dtype=complex, format="csr"),
            dims=[[2] * num_qubits, [2] * num_qubits],
        )

class TOFFOLI(ControlledGate):
    """
    TOFFOLI gate.

    Examples
    --------
    >>> from qutip_qip.operations import TOFFOLI
    >>> TOFFOLI([0, 1, 2]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
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
    target_gate = X

    num_qubits: int = 3
    num_ctrl_qubits: int = 2

    @staticmethod
    def get_qobj() -> Qobj:
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
    >>> FREDKIN([0, 1, 2]).get_qobj() # doctest: +NORMALIZE_WHITESPACE
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
    target_gate = SWAP

    num_qubits: int = 3
    num_ctrl_qubits: int = 1

    @staticmethod
    def get_qobj() -> Qobj:
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
