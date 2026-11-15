from __future__ import annotations
from collections import ChainMap
from collections.abc import Callable
from dataclasses import dataclass, field

from qutip_qip.operations.bloq import BloqBuilder, Bloq
from qutip_qip.operations.op import Op, OpInstruction, get_op_class

# The implementation of an Op is just a function
# which takes a BloqBuilder instances and adds the relevant gates.
BloqImpl = Callable[..., None]


@dataclass(eq=False)
class BloqRepository:
    """
    Maps Op to its implementation. The actual implemented is lazy and only constructed during the runtime.
    """

    # The entries are mapped using Op hashes of the Op type since Op carries params and the
    # registry does not care about them e.g. MultiSWAP(3), MultiSWAP(4) are two different Op values
    # but share the same implementation.

    _entries: ChainMap[Op | type[Op], BloqImpl] = field(
        default_factory=ChainMap, init=False
    )

    def register(
        self,
        op: Op | type[Op],
        impl: BloqImpl | None = None,
        replace: bool = False,
    ):
        if impl is None:
            return lambda f: self.register(op, f, replace=replace)

        key = get_op_class(op)
        if key is None:
            raise TypeError(f"Register the Op family, got {op!r}.")

        if key in self._entries.maps[0] and not replace:
            raise ValueError(
                f"{key.name} is already registered and replace argument is False."
            )

        self._entries[key] = impl
        return impl

    def resolve(self, op: Op | type[Op]) -> Bloq | None:
        impl = self._entries.get(get_op_class(op))
        if impl is None:
            return None

        # Constructing the bloq for the given params
        builder = BloqBuilder(op.num_qreg, op.num_creg, op.qreg_dim)
        impl(op, builder)
        return builder.to_bloq()

    def scope(self) -> BloqRepository:
        """
        creates a child repository that sees everything the parent has,
        but whose own registrations stay private to it.
        """
        child_repository = BloqRepository()

        # New chainmap with an empty child [<new empty dict>, *parent.maps]
        # The parent's entries are shared by reference, not copied.
        child_repository._entries = self._entries.new_child()
        return child_repository

    def __contains__(self, op: Op | type[Op]) -> bool:
        return get_op_class(op) in self._entries


DEFAULT_REPOSITORY = BloqRepository()


def register_op(
    op: Op | type[Op],
    impl: BloqImpl | None = None,
    *,
    repository: BloqRepository = DEFAULT_REPOSITORY,
    replace: bool = False,
):
    return repository.register(op, impl, replace=replace)


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

    # Without this, RX -> RY -> RX would recurse forever when both are missing.
    if op in expanded:
        raise ValueError(
            f"{op!r} decomposes back into itself, it cannot reach the primitives."
        )

    bloq = repository.resolve(op)
    if bloq is None:
        raise NotImplementedError(f"{op!r} has no decomposition in this repository.")
    builder.add_global_phase(bloq.global_phase)

    aux_creg_end = aux_creg_start + bloq.num_aux_creg
    missing = aux_creg_end - builder.num_aux_creg
    if missing > 0:
        builder.add_aux_creg(missing)

    aux_creg = builder.aux_creg[aux_creg_start:aux_creg_end]
    aux_qreg = tuple(
        q for dim in bloq.aux_qreg_dim for q in builder.add_aux_qreg(dim=dim)
    )

    qreg_map = instruction.qreg + aux_qreg
    creg_map = instruction.creg + aux_creg

    for ins in bloq.instructions:
        placed = OpInstruction(
            ins.op,
            tuple(qreg_map[q] for q in ins.qreg),
            tuple(creg_map[c] for c in ins.creg),
            ins.style,
        )
        _expand(placed, builder, repository, primitives, expanded | {op}, aux_creg_end)
    builder.release_aux_qreg(aux_qreg)


def compile(
    op: Op, repository: BloqRepository, primitives: frozenset[Op | type[Op]]
) -> Bloq:
    builder = BloqBuilder(op.num_qreg, op.num_creg, op.qreg_dim)
    _expand(
        OpInstruction(op, builder.qreg, builder.creg),
        builder,
        repository,
        primitives,
        frozenset(),
        0,
    )
    return builder.to_bloq()
