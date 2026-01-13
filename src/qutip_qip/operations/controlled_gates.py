from functools import partial

from qutip import Qobj
from qutip_qip.operations import (
    Gate,
    TwoQubitGate,
    cphase,
    controlled_gate,
    X,
    Y,
    Z,
    S,
    T,
    RX,
    RY,
    RZ,
)

class ControlledGate(Gate):
    def __init__(
        self,
        target_gate,
        controls,
        targets,
        control_value,
        **kwargs,
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


class MultiControlledGate(Gate):
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
        return cphase(self.arg_value).tidyup()

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
