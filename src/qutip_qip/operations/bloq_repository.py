from __future__ import annotations
from collections import ChainMap
from collections.abc import Callable
from dataclasses import dataclass, field

from qutip_qip.operations.bloq import BloqBuilder, Bloq
from qutip_qip.operations.op import Op, get_op_class

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

    _entries: ChainMap[Op | type[Op], tuple[BloqImpl, bool]] = field(
        default_factory=ChainMap, init=False
    )

    def register(
        self,
        op: Op | type[Op],
        impl: BloqImpl | None = None,
        replace: bool = False,
        resolve: bool = False,
    ):
        if impl is None:
            return lambda f: self.register(op, f, replace=replace, resolve=resolve)

        key = get_op_class(op)
        if key is None:
            raise TypeError(f"Register the Op family, got {op!r}.")

        if key in self._entries.maps[0] and not replace:
            raise ValueError(
                f"{key.name} is already registered and replace argument is False."
            )

        self._entries[key] = (impl, resolve)
        return impl

    def _lookup(self, op: Op) -> tuple[BloqImpl, bool] | None:
        for key in (op, *type(op).__mro__):
            if key in self._entries:
                return self._entries[key]
        return None

    def resolve(self, op: Op | type[Op]) -> Bloq:
        entry = self._lookup(op)
        if entry is None:
            raise NotImplementedError(
                f"{op!r} has no decomposition in this repository."
            )

        # Constructing the bloq for the given params
        impl, requires_resolve = entry
        builder = BloqBuilder(op.num_qreg, op.num_creg, op.qreg_dim)

        if requires_resolve:
            impl(op, builder, self.resolve)
        else:
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
        return self._lookup(op) is not None


DEFAULT_REPOSITORY = BloqRepository()


def register_op(
    op: Op | type[Op],
    impl: BloqImpl | None = None,
    *,
    repository: BloqRepository = DEFAULT_REPOSITORY,
    replace: bool = False,
    resolve: bool = False,
):
    return repository.register(op, impl, replace=replace, resolve=resolve)
