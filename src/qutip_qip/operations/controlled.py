import inspect
import math
from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy.linalg import block_diag

from qutip import Qobj
from qutip_qip.operations.gateclass import Gate
from qutip_qip.operations.helper import class_or_instance_method
from qutip_qip.operations.op import Op, op_factory
from qutip_qip.operations.modifier import Modifier
from qutip_qip.typing import Int


class Controlled(Modifier):
    """
    Abstract class for defining a Controlled Op.
    """

    config = Op.config | {"num_ctrl_qreg", "ctrl_dim", "ctrl_value", "target"}
    num_ctrl_qreg: ClassVar[int] = 1
    ctrl_value: ClassVar[int | None] = (
        None  # None: all controls in |1> (|d-1> for control on a qudit)
    )
    ctrl_dim: ClassVar[tuple[int, ...] | None] = None  # None: qubits

    _derived = {
        "num_qreg": lambda c, t: c.num_ctrl_qreg + t.num_qreg,
        "qreg_dim": lambda c, t: c.ctrl_dim + t.qreg_dim,
        "num_creg": lambda c, t: t.num_creg,
        "self_inverse": lambda c, t: t.self_inverse,
    }

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            num_ctrl_qreg = cls.num_ctrl_qreg
            if not isinstance(num_ctrl_qreg, Int) or num_ctrl_qreg < 1:
                raise ValueError(
                    f"{cls.__name__}: num_ctrl_qreg must be an integer >= 1."
                )

            # Say if the Op we were controlling was itself controlled e.g. CX
            # Then inheriting target's ctrl_dim, ctrl_value is wrong which happens during MRO.
            # Instead we set the values retrieved from target as None and then set them in next step.
            if vars(cls).keys() & {"num_ctrl_qreg", "ctrl_dim"}:
                for attr in ("ctrl_dim", "ctrl_value"):
                    if attr not in vars(cls):
                        setattr(cls, attr, None)

            # Default value and validation for ctrl_dim
            if cls.ctrl_dim is None:
                cls.ctrl_dim = (2,) * num_ctrl_qreg

            if len(cls.ctrl_dim) != num_ctrl_qreg or any(
                not isinstance(d, Int) or d < 2 for d in cls.ctrl_dim
            ):
                raise ValueError(
                    f"{cls.__name__}: ctrl_dim={cls.ctrl_dim} needs {num_ctrl_qreg} entries, each >= 2."
                )

            # Default value and validation for ctrl_value
            size = math.prod(cls.ctrl_dim)
            if cls.ctrl_value is None:
                cls.ctrl_value = size - 1

            if not isinstance(cls.ctrl_value, Int) or not 0 <= cls.ctrl_value < size:
                raise ValueError(f"{cls.__name__}: ctrl_value must be in [0, {size}).")

            # Acts as hash (identity) for a Controlled Op
            cls._key = (cls.target, num_ctrl_qreg, cls.ctrl_value, cls.ctrl_dim)

        super().__init_subclass__()

        # For a derived target e.g. MCX(n) only knows num_qreg per instance, validate_params covers it.
        # Below check is for a non-derived target
        if (
            not inspect.isabstract(cls)
            and not cls.target.is_derived
            and cls.target.num_qreg < 1
        ):
            raise TypeError(
                f"{cls.__name__}: target must act on at least one register."
            )

    def validate_params(self) -> None:
        # Run the validate_params method by buidling the target
        # For non-parametric target Op like X, validate_params is just pass be default, so it passes
        # For Parametric target Op (say RX), when we call CRX(theta), RX(theta) gets constructed which validates the params
        if self.target_op().num_qreg < 1:
            raise ValueError(f"{self.name}: target must act on at least one register.")


class ControlledGate(Controlled, Gate):
    r"""
    Controlled Op whose target is a Gate.

    Attributes
    ----------
    target : :class:`.Gate`
        The gate to be applied to the target qubits.

    num_ctrl_qreg : int
        The number of qubits acting as controls.

    ctrl_value : int
        The decimal value of the control state required to execute the
        unitary operator on the target qubits.

        Example:
        If the gate should execute when the 0-th qubit is $|1\rangle$ set ``ctrl_value=1``.
        If the gate should execute when two control qubits are $|10\rangle$ (binary 10), set ``ctrl_value=0b10``.
    """

    config = Controlled.config | Gate.config

    def __init_subclass__(cls) -> None:
        """
        Validates the subclass definition.
        """

        super().__init_subclass__()
        if inspect.isabstract(cls):
            return

        # Target must be a Gate class
        if not issubclass(cls.target, Gate):
            raise TypeError(f"{cls.__name__}: target must be a Gate.")

    @class_or_instance_method
    def compute_qobj(cls_ins, dtype: str = "dense") -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        return controlled_gate_unitary(
            U=cls_ins.target_op().get_qobj(dtype),
            num_ctrls=cls_ins.num_ctrl_qreg,
            ctrl_value=cls_ins.ctrl_value,
            ctrl_dim=cls_ins.ctrl_dim,
            dtype=dtype,
        )

    @class_or_instance_method
    def inverse(cls_ins) -> Gate | type[Gate]:
        if cls_ins.self_inverse:
            return cls_ins

        # We invert the target gate
        inverse_gate = cls_ins.target_op().inverse()
        gate_type = (
            inverse_gate if isinstance(inverse_gate, type) else type(inverse_gate)
        )

        # Same gate_type, changed params e.g. in case of RX(0.5)
        if gate_type is cls_ins.target:
            if isinstance(cls_ins, type):
                return cls_ins
            return type(cls_ins)(*inverse_gate.params)

        cgate = controlled(
            gate_type, cls_ins.num_ctrl_qreg, cls_ins.ctrl_value, cls_ins.ctrl_dim
        )
        return cgate if isinstance(inverse_gate, type) else cgate(*inverse_gate.params)


@op_factory
@lru_cache(maxsize=128)
def controlled(
    target: type[Op],
    num_ctrl_qreg: int = 1,
    ctrl_value: int | None = None,
    ctrl_dim: tuple[int, ...] | None = None,
    name: str | None = None,
) -> type[Controlled]:

    if not isinstance(target, type) and not issubclass(target, Op):
        raise TypeError(
            f"controlled() takes an op class, e.g. controlled(RX), got {target!r}."
        )

    # What are we controlling a Gate or an Op
    base_class = ControlledGate if issubclass(target, Gate) else Controlled

    # A class body cannot read a parameter if what it assigns has the same name, so we use aliases.
    _name = name or f"{'C' * num_ctrl_qreg}{target.name}"
    _target, _num_ctrl_qreg, _ctrl_value, _ctrl_dim = (
        target,
        num_ctrl_qreg,
        ctrl_value,
        ctrl_dim,
    )

    class _Controlled(base_class):
        name = _name
        target = _target
        num_ctrl_qreg = _num_ctrl_qreg
        ctrl_value = _ctrl_value
        ctrl_dim = _ctrl_dim

    # Without this every factory-made class is called _Controlled when cls.__name__ is called
    _Controlled.__name__ = _Controlled.__qualname__ = _name
    return _Controlled


def controlled_gate_unitary(
    U: Qobj,
    num_ctrls: int,
    ctrl_value: int,
    ctrl_dim: tuple[int, ...] | None = None,
    dtype: str = "dense",
) -> Qobj:
    """
    Create an N-qubit controlled gate from a single-qubit gate U with the given
    control and target qubits.

    Parameters
    ----------
    U : :class:`qutip.Qobj`
        An arbitrary unitary gate.
    num_ctrls : int
        The total number of control qubits.
    control_value : int
        The decimal value of the controlled qubits that activates the gate U.
    ctrl_dim: tuple[int, ...]
        For generalized qudit targets

    Returns
    -------
    result : qobj
        Quantum object representing the controlled-U gate.
    """
    if ctrl_dim is None:
        ctrl_dim = (2,) * num_ctrls

    blocks = [np.eye(U.shape[0]) for _ in range(math.prod(ctrl_dim))]
    blocks[ctrl_value] = U.full()
    return Qobj(block_diag(*blocks), dims=[list(ctrl_dim) + U.dims[0]] * 2, dtype=dtype)
