from qutip import Qobj

from qutip_qip.operations.bloq import Bloq, BloqBuilder
from qutip_qip.operations.bloq_repository import register_op
from qutip_qip.operations.controlled import Controlled
from qutip_qip.operations.inverse import Inverse
from qutip_qip.operations.unitary import Unitary
from qutip_qip.operations.gates import PHASE, TOFFOLI, X
from qutip_qip.typing import Int


def _from_op(attr: str) -> property:
    return property(lambda self: getattr(self.op, attr))


def _reserve_aux(builder: BloqBuilder, bloq: Bloq) -> None:
    for dim in bloq.aux_qreg_dim:
        builder.add_aux_qreg(dim=dim)
    if bloq.num_aux_creg:
        builder.add_aux_creg(bloq.num_aux_creg)


class Power(Unitary):
    op: Unitary
    k: int

    locals().update(
        {
            a: _from_op(a)
            for a in (
                "num_qreg",
                "qreg_dim",
                "self_inverse",
                "is_clifford",
                "has_matrix",
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


@register_op(Inverse, resolve=True)
def _inverse(op, bloq, resolve):
    inner = resolve(op.target_unitary)
    _reserve_aux(bloq, inner)

    bloq.add_global_phase(-inner.global_phase)
    for ins in reversed(inner.instructions):
        bloq.add(Inverse(ins.op), ins.qreg, ins.creg)


@register_op(Controlled, resolve=True)
def _controlled(op, bloq, resolve):

    # TODO: expand this controlled resolver, to allow for qudit resolving
    # and different ways of resolving controlled (with and without ancilla), measurement
    # Ideally this function can become a dispatch, the option can be chosen at compile time.
    if any(dim != 2 for dim in op.ctrl_dim):
        raise NotImplementedError(f"{op!r}: no generic rule for qudit controls.")

    num_ctrl, target = op.num_ctrl_qreg, op.target_unitary
    zeros = [c for c in range(num_ctrl) if not op.ctrl_value >> (num_ctrl - 1 - c) & 1]

    if zeros:
        for q in zeros:
            bloq.add(X, q)
        bloq.add(Controlled(target, num_ctrl), bloq.qreg, bloq.creg)
        for q in zeros:
            bloq.add(X, q)

    elif num_ctrl > 1:
        (aux,) = bloq.add_aux_qreg()
        bloq.add(TOFFOLI, (0, 1, aux))
        bloq.add(
            Controlled(target, num_ctrl - 1), (aux, *range(2, op.num_qreg)), bloq.creg
        )
        bloq.add(TOFFOLI, (0, 1, aux))

    else:
        inner = resolve(target)
        _reserve_aux(bloq, inner)

        if inner.global_phase:  # a relative phase once controlled
            bloq.add(PHASE(inner.global_phase), 0)
        for ins in inner.instructions:
            bloq.add(Controlled(ins.op), (0, *(q + 1 for q in ins.qreg)), ins.creg)
