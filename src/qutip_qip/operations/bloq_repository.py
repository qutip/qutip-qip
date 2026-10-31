from __future__ import annotations
from collections import ChainMap
from collections.abc import Callable
from dataclasses import dataclass, field

from qutip_qip.operations.bloq import BloqBuilder, Bloq
from qutip_qip.operations.op import Op

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

    _entries: ChainMap[type[Op], BloqImpl] = field(default_factory=ChainMap, init=False)

    def register(
        self,
        op: Op | type[Op],
        impl: BloqImpl | None = None,
        replace: bool = False,
    ):
        if impl is None:
            return lambda f: self.register(op, f, replace=replace)

        if not (isinstance(op, type) and issubclass(op, Op)):
            raise TypeError(f"Register the Op class, not a value, got {op!r}.")

        if op in self._entries.maps[0] and not replace:
            raise ValueError(
                f"{op.name!r} is already registered and replace argument is False."
            )

        self._entries[op] = impl
        return impl

    def resolve(self, op: Op | type[Op]) -> Bloq | None:
        if isinstance(op, Op):
            impl = self._entries.get(hash(type(op)))
        else:
            impl = self._entries.get(hash(op))

        if impl is None:
            return None

        # Constructing the bloq for the given params
        builder = BloqBuilder(op.num_qreg, op.num_creg, op.qreg_dim)
        impl(op, builder)
        return builder.build()

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
        return (op if isinstance(op, type) else type(op)) in self._entries


DEFAULT_REPOSITORY = BloqRepository()


def register_op(
    op: type[Op],
    impl: BloqImpl | None = None,
    *,
    repository: BloqRepository = DEFAULT_REPOSITORY,
    replace: bool = False,
):
    return repository.register(op, impl, replace=replace)
