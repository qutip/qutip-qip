import numbers
from collections.abc import Iterable
from itertools import product, chain
from functools import partial, reduce
from operator import mul
from typing import Optional

import warnings
import inspect
from copy import deepcopy

import numpy as np
import scipy.sparse as sp

import qutip
from qutip import Qobj, identity, qeye, sigmax, sigmay, sigmaz, tensor, fock_dm
from .gates import (
    rx,
    ry,
    rz,
    sqrtnot,
    snot,
    phasegate,
    x_gate,
    y_gate,
    z_gate,
    cy_gate,
    cz_gate,
    s_gate,
    t_gate,
    cs_gate,
    qrot,
    qasmu_gate,
    ct_gate,
    cphase,
    cnot,
    csign,
    berkeley,
    swapalpha,
    molmer_sorensen,
    swap,
    iswap,
    sqrtswap,
    sqrtiswap,
    fredkin,
    toffoli,
    controlled_gate,
    globalphase,
    expand_operator,
)

from packaging.version import parse as parse_version

if parse_version(qutip.__version__) >= parse_version("5.dev"):
    is_qutip5 = True
else:
    is_qutip5 = False

__all__ = [
    "Gate",
    "GATE_CLASS_MAP",
    "X",
    "Y",
    "Z",
    "RX",
    "RY",
    "RZ",
    "H",
    "SNOT",
    "SQRTNOT",
    "S",
    "T",
    "R",
    "QASMU",
    "SWAP",
    "ISWAP",
    "CNOT",
    "SQRTSWAP",
    "SQRTISWAP",
    "SWAPALPHA",
    "MS",
    "TOFFOLI",
    "FREDKIN",
    "BERKELEY",
    "CNOT",
    "CSIGN",
    "CRX",
    "CRY",
    "CRZ",
    "CY",
    "CX",
    "CZ",
    "CS",
    "CT",
    "CPHASE",
    "RZX",
]

"""
.. testsetup::

   import numpy as np
   np.set_printoptions(5)
"""


class Gate:
    r"""
    Base class for a quantum gate,
    concrete gate classes need to be defined as subclasses.

    Parameters
    ----------
    targets : list or int
        The target qubits fo the gate.
    controls : list or int
        The controlling qubits of the gate.
    arg_value : object
        Argument value of the gate. It will be saved as an attributes and
        can be accessed when generating the `:obj:qutip.Qobj`.
    classical_controls : int or list of int, optional
        Indices of classical bits to control the unitary operator.
    control_value : int, optional
        The decimal value of controlling bits for executing
        the unitary operator on the target qubits.
        E.g. if the gate should be executed when the zero-th bit is 1,
        ``controll_value=1``;
        If the gate should be executed when the two bits are 1 and 0,
        ``controll_value=2``.
    classical_control_value : int, optional
        The decimal value of controlling classical bits for executing
        the unitary operator on the target qubits.
        E.g. if the gate should be executed when the zero-th bit is 1,
        ``controll_value=1``;
        If the gate should be executed when the two bits are 1 and 0,
        ``controll_value=2``.
        The default is ``2**len(classical_controls)-1``
        (i.e. all classical controls are 1).
    arg_label : string
        Label for the argument, it will be shown in the circuit plot,
        representing the argument value provided to the gate, e.g,
        if ``arg_label="\phi" the latex name for the gate in the circuit plot
        will be ``$U(\phi)$``.
    name : string, optional
        The name of the gate. This is kept for backward compatibility
        to identify different gates.
        In most cases it is identical to the class name,
        but that is not guaranteed.
        It is recommended to use ``isinstance``
        or ``issubclass`` to identify a gate rather than
        comparing the name string.
    """

    def __init__(
        self,
        name=None,
        targets=None,
        controls=None,
        arg_value=None,
        control_value=None,
        classical_controls=None,
        classical_control_value: Optional[int] = None,
        arg_label=None,
        **kwargs,
    ):
        """
        Create a gate with specified parameters.
        """

        self.name = name if name is not None else self.__class__.__name__
        self.targets = None
        self.controls = None
        self.classical_controls = None

        if not isinstance(targets, Iterable) and targets is not None:
            self.targets = [targets]
        else:
            self.targets = targets

        if not isinstance(controls, Iterable) and controls is not None:
            self.controls = [controls]
        else:
            self.controls = controls

        if (
            not isinstance(classical_controls, Iterable)
            and classical_controls is not None
        ):
            self.classical_controls = [classical_controls]
        else:
            self.classical_controls = classical_controls

        if (
            self.classical_controls is not None
            and classical_control_value is None
        ):
            self.classical_control_value = (
                2 ** len(self.classical_controls) - 1
            )
        else:
            self.classical_control_value = classical_control_value
        self.control_value = control_value
        self.arg_value = arg_value
        self.arg_label = arg_label
        self.latex_str = r"U"

        for ind_list in [self.targets, self.controls, self.classical_controls]:
            if ind_list is None:
                continue
            all_integer = all(
                [isinstance(ind, numbers.Integral) for ind in ind_list]
            )
            if not all_integer:
                raise ValueError("Index of a qubit must be an integer")

    def get_all_qubits(self):
        """
        Return a list of all qubits that the gate operator
        acts on.
        The list concatenates the two lists representing
        the controls and the targets qubits while retains the order.

        Returns
        -------
        targets_list : list of int
            A list of all qubits, including controls and targets.
        """
        if self.controls is not None:
            return self.controls + self.targets
        if self.targets is not None:
            return self.targets
        else:
            # Special case: the global phase gate
            return []

    def __str__(self):
        str_name = (
            "Gate(%s, targets=%s, controls=%s,"
            " classical controls=%s, control_value=%s, classical_control_value=%s)"
        ) % (
            self.name,
            self.targets,
            self.controls,
            self.classical_controls,
            self.control_value,
            self.classical_control_value,
        )
        return str_name

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)

    def _to_qasm(self, qasm_out):
        """
        Pipe output of gate signature and application to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_gate = qasm_out.qasm_name(self.name)

        if not qasm_gate:
            error_str = "{} gate's qasm defn is not specified".format(
                self.name
            )
            raise NotImplementedError(error_str)

        if self.classical_controls:
            err_msg = "Exporting controlled gates is not implemented yet."
            raise NotImplementedError(err_msg)
        else:
            qasm_out.output(
                qasm_out._qasm_str(
                    qasm_gate, self.controls, self.targets, self.arg_value
                )
            )

    def get_compact_qobj(self):
        """
        Get the compact :class:`qutip.Qobj` representation of the gate
        operator, ignoring the controls and targets.
        In the unitary representation,
        it always assumes that the first few qubits are controls,
        then targets.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        # TODO This will be moved to each sub-class of Gate.
        # However, one first needs to replace the direct use of Gate in
        # other modules.
        if self.name == "RX":
            qobj = rx(self.arg_value)
        elif self.name == "RY":
            qobj = ry(self.arg_value)
        elif self.name == "RZ":
            qobj = rz(self.arg_value)
        elif self.name == "X":
            qobj = x_gate()
        elif self.name == "Y":
            qobj = y_gate()
        elif self.name == "CY":
            qobj = cy_gate()
        elif self.name == "Z":
            qobj = z_gate()
        elif self.name == "CZ":
            qobj = cz_gate()
        elif self.name == "T":
            qobj = t_gate()
        elif self.name == "CT":
            qobj = ct_gate()
        elif self.name == "S":
            qobj = s_gate()
        elif self.name == "CS":
            qobj = cs_gate()
        elif self.name == "SQRTNOT":
            qobj = sqrtnot()
        elif self.name == "SNOT":
            qobj = snot()
        elif self.name == "PHASEGATE":
            qobj = phasegate(self.arg_value)
        elif self.name == "R":
            qobj = qrot(*self.arg_value)
        elif self.name == "QASMU":
            qobj = qasmu_gate(self.arg_value)
        elif self.name == "CRX":
            qobj = controlled_gate(rx(self.arg_value))
        elif self.name == "CRY":
            qobj = controlled_gate(ry(self.arg_value))
        elif self.name == "CRZ":
            qobj = controlled_gate(rz(self.arg_value))
        elif self.name == "CPHASE":
            qobj = cphase(self.arg_value)
        elif self.name == "CNOT":
            qobj = cnot()
        elif self.name == "CSIGN":
            qobj = csign()
        elif self.name == "BERKELEY":
            qobj = berkeley()
        elif self.name == "SWAPalpha":
            qobj = swapalpha(self.arg_value)
        elif self.name == "SWAP":
            qobj = swap()
        elif self.name == "ISWAP":
            qobj = iswap()
        elif self.name == "SQRTSWAP":
            qobj = sqrtswap()
        elif self.name == "SQRTISWAP":
            qobj = sqrtiswap()
        elif self.name == "FREDKIN":
            qobj = fredkin()
        elif self.name == "TOFFOLI":
            qobj = toffoli()
        elif self.name == "IDLE":
            qobj = qeye(2)
        elif self.name == "GLOBALPHASE":
            raise NotImplementedError(
                "Globalphase gate has no compack qobj representation."
            )
        else:
            raise NotImplementedError(f"{self.name} is an unknown gate.")
        return qobj

    def get_qobj(self, num_qubits=None, dims=None):
        """
        Get the :class:`qutip.Qobj` representation of the gate operator.
        The operator is expanded to the full Herbert space according to
        the controls and targets qubits defined for the gate.

        Parameters
        ----------
        num_qubits : int, optional
            The number of qubits.
            If not given, use the minimal number of qubits required
            by the target and control qubits.
        dims : list, optional
            A list representing the dimensions of each quantum system.
            If not given, it is assumed to be an all-qubit system.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        if self.name == "GLOBALPHASE":
            if num_qubits is not None:
                return globalphase(self.arg_value, num_qubits)
            else:
                raise ValueError(
                    "The number of qubits must be provided for "
                    "global phase gates."
                )

        all_targets = self.get_all_qubits()
        if num_qubits is None:
            num_qubits = max(all_targets) + 1
        return expand_operator(
            self.get_compact_qobj(),
            dims=dims,
            targets=all_targets,
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
        return qutip.sigmax() if not is_qutip5 else qutip.sigmax(dtype="dense")


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
        return qutip.sigmay() if not is_qutip5 else qutip.sigmay(dtype="dense")


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
        return qutip.sigmaz() if not is_qutip5 else qutip.sigmaz(dtype="dense")


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


class TwoQubitGate(Gate):
    """Abstract two-qubit gate."""

    def __init__(self, targets, **kwargs):
        super().__init__(targets=targets, **kwargs)
        if len(self.get_all_qubits()) != 2:
            raise ValueError(
                f"Gate {self.__class__.__name__} requires two targets"
            )


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
        return swap()


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
        return iswap()


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
        return sqrtswap()


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
        return sqrtiswap()


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
        return berkeley()


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
        return swapalpha(self.arg_value)


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
        return molmer_sorensen(*self.arg_value)


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
        return toffoli()


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
        return fredkin()


class ControlledGate(Gate):
    def __init__(
        self, controls, targets, control_value, target_gate, **kwargs
    ):
        super().__init__(
            controls=controls,
            targets=targets,
            control_value=control_value,
            target_gate=target_gate,
            **kwargs,
        )
        self.controls = (
            [controls] if not isinstance(controls, list) else controls
        )
        self.control_value = control_value
        self.target_gate = target_gate
        self.kwargs = kwargs
        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        self.latex_str = target_gate(
            targets=self.targets, **self.kwargs
        ).latex_str

    def get_compact_qobj(self):
        return controlled_gate(
            U=self.target_gate(
                targets=self.targets, **self.kwargs
            ).get_compact_qobj(),
            controls=list(range(len(self.controls))),
            targets=list(
                range(
                    len(self.controls), len(self.targets) + len(self.controls)
                )
            ),
            control_value=self.control_value,
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
        return cnot()


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
        return csign()


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
        return csign()


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
        return cphase(self.arg_value).tidyup()


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

GATE_CLASS_MAP = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "RX": RX,
    "RY": RY,
    "RZ": RZ,
    "H": H,
    "SNOT": SNOT,
    "SQRTNOT": SQRTNOT,
    "S": S,
    "T": T,
    "R": R,
    "QASMU": QASMU,
    "SWAP": SWAP,
    "ISWAP": ISWAP,
    "iSWAP": ISWAP,
    "CNOT": CNOT,
    "SQRTSWAP": SQRTSWAP,
    "SQRTISWAP": SQRTISWAP,
    "SWAPALPHA": SWAPALPHA,
    "SWAPalpha": SWAPALPHA,
    "BERKELEY": BERKELEY,
    "MS": MS,
    "TOFFOLI": TOFFOLI,
    "FREDKIN": FREDKIN,
    "CNOT": CNOT,
    "CSIGN": CSIGN,
    "CRX": CRX,
    "CRY": CRY,
    "CRZ": CRZ,
    "CY": CY,
    "CX": CX,
    "CZ": CZ,
    "CS": CS,
    "CT": CT,
    "CPHASE": CPHASE,
    "RZX": RZX,
}
