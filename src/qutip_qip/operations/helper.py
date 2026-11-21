from typing import ClassVar, Self

from qutip import Qobj

from qutip_qip.operations.bloq_repository import register_op
from qutip_qip.operations.inverse import Inverse
from qutip_qip.operations.unitary import (
    Modifier,
    Unitary,
    UnitaryFamily,
    get_unitary_class,
)
from qutip_qip.typing import Int


class Power(Modifier):
    _settings = ("exponent",)
    config = Modifier.config | set(_settings)
    exponent: ClassVar[int]

    @classmethod
    def _apply_modifier(cls, op: UnitaryFamily, exponent: int) -> Self | type[Self]:
        family = get_unitary_class(op)
        if isinstance(exponent, bool) or not isinstance(exponent, Int) or exponent < 0:
            raise ValueError(
                f"Power: exponent must be an integer >= 0, got {exponent!r}."
            )
        return cls._make_modifier(
            op, f"{family.__name__}^{exponent}", exponent=exponent
        )

    @property
    def name(self) -> str:
        return f"{self.target.name}^{self.exponent}"

    @property
    def latex_str(self) -> str:
        return rf"{{{self.target.latex_str}}}^{{{self.exponent}}}"

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return self.target.get_qobj(dtype) ** self.exponent

    def inverse(self) -> Unitary:
        """
        The inverse of a power keeps the same exponent and inverts only the
        target. For example, the inverse of Power(T, 3) is Power(Tdag, 3).
        """
        return Power(Inverse(self.target), self.exponent)

    def __repr__(self) -> str:
        return f"{self.target!r}^{self.exponent}"


@register_op(Power)
def _power(p, bloq):
    for _ in range(p.exponent):
        bloq.add(p.target, tuple(range(p.num_qreg)))


# Add SELECT
