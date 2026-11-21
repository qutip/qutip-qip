from qutip import Qobj

from qutip_qip.operations.bloq_repository import register_op
from qutip_qip.operations.inverse import Inverse
from qutip_qip.operations.unitary import Modifier, Unitary

from qutip_qip.typing import Int


def _from_op(attr: str) -> property:
    return property(lambda self: getattr(self.op, attr))


class Power(Modifier):
    op: Unitary
    k: int

    target = property(lambda self: self.op)

    locals().update(
        {
            a: _from_op(a)
            for a in (
                "num_qreg",
                "qreg_dim",
                "self_inverse",
                "is_clifford",
            )
        }
    )
    name = property(lambda self: f"{self.op.name}^{self.k}")
    latex_str = property(lambda self: rf"{self.op.latex_str}^{{{self.k}}}")

    def validate_params(self) -> None:
        if not isinstance(self.op, Unitary):
            raise TypeError(f"Power takes a unitary op, got {self.op!r}.")
        if isinstance(self.k, bool) or not isinstance(self.k, Int) or self.k < 0:
            raise ValueError(f"Power: k must be an integer >= 0, got {self.k!r}.")

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return self.op.get_qobj(dtype) ** self.k

    def inverse(self) -> Unitary:
        return Power(Inverse(self.op), self.k)

    def __repr__(self) -> str:
        return f"{self.op!r}^{self.k}"


@register_op(Power)
def _power(p, bloq):
    for _ in range(p.k):
        bloq.add(p.op, tuple(range(p.num_qreg)))


# Add SELECT
