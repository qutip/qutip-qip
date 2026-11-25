from qutip_qip.algorithms.qft import QFT
from qutip_qip.operations import Inverse, Power, Unitary, Controlled, register_op
from qutip_qip.operations.gates import H
from qutip_qip.typing import Int


class QPE(Unitary):
    unitary: Unitary
    num_counting_qreg: int

    @property
    def num_qreg(self) -> int:
        return self.num_counting_qreg + self.unitary.num_qreg

    @property
    def qreg_dim(self) -> tuple[int, ...]:
        return (2,) * self.num_counting_qreg + self.unitary.qreg_dim

    def validate_params(self) -> None:
        if not isinstance(self.unitary, Unitary):
            raise TypeError(f"QPE takes a unitary op, got {self.unitary!r}.")
        t = self.num_counting_qreg
        if isinstance(t, bool) or not isinstance(t, Int) or t < 1:
            raise ValueError(
                f"QPE: num_counting_qreg must be an integer >= 1, got {t!r}."
            )


@register_op(QPE)
def _qpe(op, bloq):
    t = op.num_counting_qreg
    target = tuple(range(t, op.num_qreg))

    for q in range(t):
        bloq.add(H, q)

    for j in range(t):
        bloq.add(Controlled(Power(op.unitary, 2 ** (t - 1 - j))), (j, *target))

    bloq.add(Inverse(QFT(t)), tuple(range(t)))
