import qutip
from qutip_qip.operations import (
    Gate,
    rx,
    ry,
    rz,
    sqrtnot,
    snot,
    s_gate,
    t_gate,
    qrot,
    qasmu_gate,
)

class SingleQubitGate(Gate):
    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        if self.targets is None or len(self.targets) != 1:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires one target"
            )
        if self.controls:
            raise ValueError(
                f"Gate {self.__class__.__name__} cannot have a control"
            )
        
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
        return qutip.sigmax(dtype="dense")


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
        return qutip.sigmay(dtype="dense")


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
        return qutip.sigmaz(dtype="dense")


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
        return rx(self.arg_value)


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
        return ry(self.arg_value)


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
        return rz(self.arg_value)


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
        return snot()


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
        return sqrtnot()


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
        return s_gate()


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
        return t_gate()


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
        return qrot(*self.arg_value)


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
        return qasmu_gate(self.arg_value)
