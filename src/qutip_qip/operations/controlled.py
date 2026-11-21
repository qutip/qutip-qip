import math
from typing import ClassVar, Self

import numpy as np
from scipy.linalg import block_diag

from qutip import Qobj
from qutip_qip.operations.inverse import Inverse
from qutip_qip.operations.op import get_op_class
from qutip_qip.operations.unitary import (
    Modifier,
    Unitary,
    UnitaryFamily,
    get_unitary_class,
)
from qutip_qip.typing import Int


class Controlled(Modifier):
    r"""
    A unitary that is applied only when its controls are in a chosen state.
    Controlled(op) takes a value like X, RX(0.5) or a classes like RX.

    Controlling an op that is already controlled adds to its controls, so
    the target is not a Controlled itself.

    A controlled gate can also be written as a class, with the controls as
    class attributes.

    Parameters
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

    _settings = ("num_ctrl_qreg", "ctrl_value", "ctrl_dim")
    config = Modifier.config | set(_settings)

    is_controlled: ClassVar[bool] = True
    num_ctrl_qreg: ClassVar[int] = 1

    # None means every control is in its highest state: |1> for a qubit.
    ctrl_value: ClassVar[int | None] = None

    # None means every control is a qubit.
    ctrl_dim: ClassVar[tuple[int, ...] | None] = None

    # Adding a control to a Clifford gate does not always give a Clifford gate.
    is_clifford: ClassVar[bool] = False

    def __init_subclass__(cls) -> None:
        if not isinstance(cls.target, property):
            cls.target, cls.num_ctrl_qreg, cls.ctrl_value, cls.ctrl_dim = cls._resolve(
                cls.target, cls.num_ctrl_qreg, cls.ctrl_value, cls.ctrl_dim
            )
        super().__init_subclass__()

    @classmethod
    def _resolve(
        cls,
        target: UnitaryFamily,
        num_ctrl_qreg: int,
        ctrl_value: int | None,
        ctrl_dim: tuple[int, ...] | None,
    ) -> tuple[UnitaryFamily, int, int, tuple[int, ...]]:
        """Check the target and the controls, and fill in the defaults."""

        family = get_unitary_class(target)

        if not isinstance(num_ctrl_qreg, Int) or num_ctrl_qreg < 1:
            raise ValueError(f"{cls.name}: num_ctrl_qreg must be an integer >= 1.")

        # Default value and validation for ctrl_dim
        if ctrl_dim is None:
            ctrl_dim = (2,) * num_ctrl_qreg

        if (
            not isinstance(ctrl_dim, tuple)
            or len(ctrl_dim) != num_ctrl_qreg
            or any(not isinstance(d, Int) or d < 2 for d in ctrl_dim)
        ):
            raise ValueError(
                f"{cls.name}: ctrl_dim={ctrl_dim} needs {num_ctrl_qreg} entries, each >= 2."
            )

        # Default value and validation for ctrl_value
        size = math.prod(ctrl_dim)
        if ctrl_value is None:
            ctrl_value = size - 1

        if not isinstance(ctrl_value, Int) or not 0 <= ctrl_value < size:
            raise ValueError(f"{cls.name}: ctrl_value must be in [0, {size}).")

        # If we are controlling an already Controlled e.g. CX
        # we can just use X as target and increase the num_ctrl_qreg, ctrl_dim count
        if issubclass(family, Controlled):
            ctrl_value = ctrl_value * math.prod(target.ctrl_dim) + target.ctrl_value
            ctrl_dim = ctrl_dim + target.ctrl_dim
            num_ctrl_qreg = len(ctrl_dim)
            target = target.target

        return target, num_ctrl_qreg, ctrl_value, ctrl_dim

    @classmethod
    def _apply_modifier(
        cls: Self,
        target: UnitaryFamily,
        num_ctrl_qreg: int = 1,
        ctrl_value: int | None = None,
        ctrl_dim: tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> Self | type[Self]:
        target, num_ctrl_qreg, ctrl_value, ctrl_dim = cls._resolve(
            target, num_ctrl_qreg, ctrl_value, ctrl_dim
        )

        target_family = get_unitary_class(target)
        prefix = "C" * num_ctrl_qreg

        if name is not None:
            op_name = name
        elif isinstance(target_family.name, str):
            op_name = prefix + target_family.name
        else:
            # The name of the target depends on its parameters, so ours does too.
            op_name = property(lambda self: prefix + self.target.name)

        # Pickle makes the class again from this call. The name the caller
        # chose is not a setting, so it has to be added here.
        controls = (num_ctrl_qreg, ctrl_value, ctrl_dim)
        recreate = (cls, (get_op_class(target), *controls, name))

        return cls._make_modifier(
            target,
            prefix + target_family.__name__,
            num_ctrl_qreg=num_ctrl_qreg,
            ctrl_value=ctrl_value,
            ctrl_dim=ctrl_dim,
            name=op_name,
            _factory=recreate,
        )

    @property
    def num_qreg(self) -> int:
        return self.num_ctrl_qreg + self.target.num_qreg

    @property
    def qreg_dim(self) -> tuple[int, ...]:
        return self.ctrl_dim + self.target.qreg_dim

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        U = self.target.get_qobj(dtype)
        blocks = [np.eye(U.shape[0]) for _ in range(math.prod(self.ctrl_dim))]
        blocks[self.ctrl_value] = U.full()
        return Qobj(
            block_diag(*blocks),
            dims=[list(self.ctrl_dim) + U.dims[0]] * 2,
            dtype=dtype,
        )

    def inverse(self) -> Unitary:
        """
        The inverse of a controlled op keeps the same controls and inverts
        only the target. For example, the inverse of CRX(0.5) is CRX(-0.5).
        """
        return Controlled(
            Inverse(self.target),
            self.num_ctrl_qreg,
            self.ctrl_value,
            self.ctrl_dim,
        )

    def __repr__(self) -> str:
        return f"{'C' * self.num_ctrl_qreg}{self.target!r}"
