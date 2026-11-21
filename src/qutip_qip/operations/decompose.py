from qutip_qip.operations.bloq import Bloq, BloqBuilder
from qutip_qip.operations.bloq_repository import (
    BloqImpl,
    BloqRepository,
    DEFAULT_REPOSITORY,
    register_op,
)
from qutip_qip.operations.controlled import Controlled
from qutip_qip.operations.inverse import Inverse
from qutip_qip.operations.gates import PHASE, TOFFOLI, X
from qutip_qip.operations.op import Op, OpInstruction, get_op_class


def decompose(
    bloq: Bloq, repository: BloqRepository, primitives: frozenset[Op | type[Op]]
) -> Bloq:
    builder = BloqBuilder(bloq.num_qreg, bloq.num_creg, bloq.qreg_dim)
    _inline(
        bloq,
        builder.qreg,
        builder.creg,
        builder,
        repository,
        primitives,
        frozenset(),
        0,
    )
    return builder.to_bloq()


def _expand(
    instruction: OpInstruction,
    builder: BloqBuilder,
    repository: BloqRepository,
    primitives: frozenset[Op | type[Op]],
    expanded: frozenset[Op],
    aux_creg_start: int,
) -> None:

    op = instruction.op
    if get_op_class(op) in primitives:
        builder.add(op, instruction.qreg, instruction.creg, instruction.style)
        return

    # Without this RX -> RY -> RX would recurse forever when both are missing.
    if op in expanded:
        raise ValueError(
            f"{op!r} decomposes back into itself, it cannot reach the primitives."
        )

    bloq = repository.resolve(op)
    _inline(
        bloq,
        instruction.qreg,
        instruction.creg,
        builder,
        repository,
        primitives,
        expanded | {op},
        aux_creg_start,
    )


def _inline(
    bloq: Bloq,
    qreg: tuple[int, ...],
    creg: tuple[int, ...],
    builder: BloqBuilder,
    repository: BloqRepository,
    primitives: frozenset[Op | type[Op]],
    expanded: frozenset[Op],
    aux_creg_start: int,
) -> None:

    builder.add_global_phase(bloq.global_phase)

    # Allocating the auxiliary (ancilla) qreg, creg
    # In case of creg, we can directly reuse existing aux_cregs instead of caring
    # whether they are clean, but in case of qreg they need to be clean ancilla.
    aux_creg_end = aux_creg_start + bloq.num_aux_creg
    missing = aux_creg_end - builder.num_aux_creg
    if missing > 0:
        builder.add_aux_creg(missing)

    aux_creg = builder.aux_creg[aux_creg_start:aux_creg_end]
    aux_qreg = tuple(
        q for dim in bloq.aux_qreg_dim for q in builder.add_aux_qreg(dim=dim)
    )

    qreg_map = qreg + aux_qreg
    creg_map = creg + aux_creg

    for ins in bloq.instructions:
        placed = OpInstruction(
            ins.op,
            tuple(qreg_map[q] for q in ins.qreg),
            tuple(creg_map[c] for c in ins.creg),
            ins.style,
        )
        _expand(placed, builder, repository, primitives, expanded, aux_creg_end)
    builder.release_aux_qreg(aux_qreg)


# Decomposition logic for Modifier


def _reserve_aux(builder: BloqBuilder, bloq: Bloq) -> None:
    for dim in bloq.aux_qreg_dim:
        builder.add_aux_qreg(dim=dim)
    if bloq.num_aux_creg:
        builder.add_aux_creg(bloq.num_aux_creg)


@register_op(Inverse, resolve=True)
def _inverse(op, bloq, resolve):
    inner = resolve(op.target)
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

    num_ctrl, target = op.num_ctrl_qreg, op.target
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
