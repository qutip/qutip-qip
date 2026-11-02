import inspect
import math
from functools import lru_cache
from typing import ClassVar

import numpy as np
from scipy.linalg import block_diag

from qutip import Qobj
from qutip_qip.operations.op import op_factory, Unitary
from qutip_qip.operations.modifier import Modifier
from qutip_qip.operations.utils import class_or_instance_method
from qutip_qip.typing import Int


class Controlled(Modifier):
    r"""
    Abstract class for defining a Controlled Op.

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

    config = Modifier.config | {"num_ctrl_qreg", "ctrl_dim", "ctrl_value"}
    is_controlled: ClassVar[bool] = True
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

            # Default value and validation for ctrl_dim
            if cls.ctrl_dim is None:
                cls.ctrl_dim = (2,) * num_ctrl_qreg

            if (
                not isinstance(cls.ctrl_dim, tuple)
                or len(cls.ctrl_dim) != num_ctrl_qreg
                or any(not isinstance(d, Int) or d < 2 for d in cls.ctrl_dim)
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

            # If we are controlling an already controlled Op e.g. CX
            # we can just use X as target and increase the num_ctrl_qreg, ctrl_dim count
            target = cls.target
            if issubclass(target, Controlled):
                cls.ctrl_value = (
                    cls.ctrl_value * math.prod(target.ctrl_dim) + target.ctrl_value
                )
                cls.ctrl_dim = cls.ctrl_dim + target.ctrl_dim
                cls.num_ctrl_qreg = num_ctrl_qreg = len(cls.ctrl_dim)
                cls.target = target.target

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
        # Run the validate_params method by building the target
        # For non-parametric target Op like X, validate_params is just pass be default, so it passes
        # For Parametric target Op (say RX), when we call CRX(theta), RX(theta) gets constructed which validates the params
        if self.target_unitary().num_qreg < 1:
            raise ValueError(f"{self.name}: target must act on at least one register.")

    @class_or_instance_method
    def compute_qobj(cls_ins, dtype: str = "dense") -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        U = cls_ins.target_unitary().get_qobj(dtype)
        blocks = [np.eye(U.shape[0]) for _ in range(math.prod(cls_ins.ctrl_dim))]
        blocks[cls_ins.ctrl_value] = U.full()
        return Qobj(
            block_diag(*blocks),
            dims=[list(cls_ins.ctrl_dim) + U.dims[0]] * 2,
            dtype=dtype,
        )

    @class_or_instance_method
    def inverse(cls_ins) -> Unitary | type[Unitary]:
        return controlled(
            cls_ins.target_unitary().inverse(),
            cls_ins.num_ctrl_qreg,
            cls_ins.ctrl_value,
            cls_ins.ctrl_dim,
        )


@op_factory
@lru_cache(maxsize=128)
def controlled(
    op: type[Unitary],
    num_ctrl_qreg: int = 1,
    ctrl_value: int | None = None,
    ctrl_dim: tuple[int, ...] | None = None,
    name: str | None = None,
) -> type[Controlled]:

    if not (isinstance(op, type) and issubclass(op, Unitary)):
        raise TypeError(
            f"controlled() takes an op class, e.g. controlled(RX), got {op!r}."
        )

    # A class body cannot read a parameter if what it assigns has the same name, so we use aliases.
    _name = name or f"{'C' * num_ctrl_qreg}{op.name}"
    _target, _num_ctrl_qreg, _ctrl_value, _ctrl_dim = (
        op,
        num_ctrl_qreg,
        ctrl_value,
        ctrl_dim,
    )

    class _Controlled(Controlled):
        name = _name
        target = _target
        num_ctrl_qreg = _num_ctrl_qreg
        ctrl_value = _ctrl_value
        ctrl_dim = _ctrl_dim

    # Without this every factory-made class is called _Controlled when cls.__name__ is called
    _Controlled.__name__ = _Controlled.__qualname__ = _name
    return _Controlled
