import copy
import inspect
import math
import warnings
import weakref
from abc import abstractmethod
from dataclasses import fields
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.linalg import block_diag

from qutip import Qobj
from qutip_qip.operations.op import op_factory
from qutip_qip.operations.unitary import Unitary
from qutip_qip.typing import Int


class ControlledUnitary(Unitary):
    r"""
    Abstract class for defining a Controlled Unitary.

    Attributes
    ----------
    target : :class:`.Unitary`
        The Op to be applied to the target qubits.

    num_ctrl_qreg : int
        The number of qreg acting as controls.

    ctrl_value : int
        The decimal value of the control state required to execute the
        unitary operator on the target qubits.

        Example:
        If the Op should execute when the 0-th qubit is $|1\rangle$ set ``ctrl_value=1``.
        If the Op should execute when two control qubits are $|10\rangle$ (binary 10), set ``ctrl_value=0b10``.
    """

    config = Unitary.config | {"target", "num_ctrl_qreg", "ctrl_dim", "ctrl_value"}
    is_controlled: ClassVar[bool] = True
    num_ctrl_qreg: ClassVar[int] = 1
    ctrl_value: ClassVar[int | None] = (
        None  # None: all controls in |1> (|d-1> for control on a qudit)
    )
    ctrl_dim: ClassVar[tuple[int, ...] | None] = None  # None: qubits
    _families: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    @property
    @abstractmethod
    def target(self) -> type[Unitary]: ...

    _derived = {
        "num_qreg": lambda c, t: c.num_ctrl_qreg + t.num_qreg,
        "qreg_dim": lambda c, t: c.ctrl_dim + t.qreg_dim,
        "num_creg": lambda c, t: t.num_creg,
        "self_inverse": lambda c, t: t.self_inverse,
    }

    def __init_subclass__(cls) -> None:
        if clash := vars(cls).keys() & cls._derived.keys():
            warnings.warn(
                f"{cls.__name__}: cannot set {', '.join(clash)}, derived from the target.",
            )

        # We are setting num_qreg (an abstract property in Op) using the target Unitary
        # So instead of doing inspect.isabstract, we just check on target directly
        target = inspect.getattr_static(cls, "target")
        if isinstance(target, property):
            return super().__init_subclass__()

        cls._configure(target)
        super().__init_subclass__()
        cls._families.setdefault(
            (cls.target, cls.num_ctrl_qreg, cls.ctrl_value, cls.ctrl_dim, cls.name), cls
        )

    @classmethod
    def _resolve(cls, target, num_ctrl_qreg, ctrl_value, ctrl_dim, name):
        """Validate and fill in the controls."""
        family = type(target) if isinstance(target, Unitary) else target
        if not (isinstance(family, type) and issubclass(family, Unitary)):
            raise TypeError(
                f"{name}: target must be a unitary value like X or a family like RX, got {target!r}."
            )

        if not isinstance(num_ctrl_qreg, Int) or num_ctrl_qreg < 1:
            raise ValueError(f"{name}: num_ctrl_qreg must be an integer >= 1.")

        # Default value and validation for ctrl_dim
        if ctrl_dim is None:
            ctrl_dim = (2,) * num_ctrl_qreg

        if (
            not isinstance(ctrl_dim, tuple)
            or len(ctrl_dim) != num_ctrl_qreg
            or any(not isinstance(d, Int) or d < 2 for d in ctrl_dim)
        ):
            raise ValueError(
                f"{name}: ctrl_dim={ctrl_dim} needs {num_ctrl_qreg} entries, each >= 2."
            )

        # Default value and validation for ctrl_value
        size = math.prod(ctrl_dim)
        if ctrl_value is None:
            ctrl_value = size - 1

        if not isinstance(ctrl_value, Int) or not 0 <= ctrl_value < size:
            raise ValueError(f"{name}: ctrl_value must be in [0, {size}).")

        # If we are controlling an already ControlledUnitary e.g. CX
        # we can just use X as target and increase the num_ctrl_qreg, ctrl_dim count
        if issubclass(family, ControlledUnitary):
            ctrl_value = ctrl_value * math.prod(target.ctrl_dim) + target.ctrl_value
            ctrl_dim = ctrl_dim + target.ctrl_dim
            num_ctrl_qreg = len(ctrl_dim)
            target = (
                target.target_unitary if isinstance(target, Unitary) else target.target
            )

        return target, num_ctrl_qreg, ctrl_value, ctrl_dim, name

    @classmethod
    def _family(cls, controls):
        """The family already made with these controls"""
        family = cls._families.get(controls)
        if family is None:
            return None
        return family if family.is_parametric else family._value

    @classmethod
    def _configure(cls, target: Unitary | type[Unitary]) -> None:
        cls.target, cls.num_ctrl_qreg, cls.ctrl_value, cls.ctrl_dim, _ = cls._resolve(
            target, cls.num_ctrl_qreg, cls.ctrl_value, cls.ctrl_dim, cls.name
        )
        target = cls.target

        # A derived target, e.g. MCX(n), only knows its registers per instance.
        for attr, rule in cls._derived.items():
            if isinstance(target, type) and target.is_derived:
                setattr(
                    cls,
                    attr,
                    property(lambda self, rule=rule: rule(self, self.target_unitary)),
                )
            else:
                setattr(cls, attr, rule(cls, target))

        if isinstance(target, type):
            # The target's parameters e.g. theta in RX, become parameter in a Controlled Op
            cls.__annotations__ = {f.name: f.type for f in fields(target)} | dict(
                cls.__annotations__
            )
            for f in fields(target):
                setattr(cls, f.name, copy.copy(f))

    def validate_params(self) -> None:
        # Run the validate_params method by building the target
        # For non-parametric target Op like X, validate_params is just pass be default, so it passes
        # For Parametric target Op (say RX), when we call CRX(theta), RX(theta) gets constructed which validates the params
        self.target_unitary

    @cached_property
    def target_unitary(self) -> Unitary:
        return self.target(*self.params) if self.params else self.target

    @property
    def _key(self):
        return (self.target_unitary, self.num_ctrl_qreg, self.ctrl_value, self.ctrl_dim)

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        U = self.target_unitary.get_qobj(dtype)
        blocks = [np.eye(U.shape[0]) for _ in range(math.prod(self.ctrl_dim))]
        blocks[self.ctrl_value] = U.full()
        return Qobj(
            block_diag(*blocks),
            dims=[list(self.ctrl_dim) + U.dims[0]] * 2,
            dtype=dtype,
        )

    def compute_inverse(self) -> Unitary:
        return controlled(
            self.target_unitary.inverse(),
            self.num_ctrl_qreg,
            self.ctrl_value,
            self.ctrl_dim,
        )

    def __repr__(self) -> str:
        return f"{'C' * self.num_ctrl_qreg}{self.target_unitary!r}"


@op_factory
def controlled(
    op: Unitary | type[Unitary],
    num_ctrl_qreg: int = 1,
    ctrl_value: int | None = None,
    ctrl_dim: tuple[int, ...] | None = None,
    name: str | None = None,
) -> ControlledUnitary | type[ControlledUnitary]:

    # A class body cannot read a parameter if what it assigns has the same name, so we use aliases.
    prefix = "C" * num_ctrl_qreg
    label = name or prefix + (op if isinstance(op, type) else type(op)).__name__
    _name = name or (
        prefix + op.name
        if isinstance(op.name, str)
        else property(lambda self: prefix + self.target_unitary.name)
    )

    controls = ControlledUnitary._resolve(
        op, num_ctrl_qreg, ctrl_value, ctrl_dim, _name
    )
    if (control_family := ControlledUnitary._family(controls)) is not None:
        return control_family
    _target, _num_ctrl_qreg, _ctrl_value, _ctrl_dim, _name = controls

    class _Controlled(ControlledUnitary):
        __qualname__ = label  # otherwise every factory-made class is named _Controlled
        name = _name
        target = _target
        num_ctrl_qreg = _num_ctrl_qreg
        ctrl_value = _ctrl_value
        ctrl_dim = _ctrl_dim

    return _Controlled
