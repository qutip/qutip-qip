from __future__ import annotations
from collections import ChainMap
from collections.abc import Callable
from dataclasses import dataclass, field

from qutip_qip.operations.bloq import BloqBuilder, Bloq
from qutip_qip.operations.op import Op

# The implementation of an Op is just a function
# which takes a BloqBuilder instances and adds the relevant gates.
BloqImpl = Callable[..., None]


def _key(op):
    # A parametric op registers by family, MultiSWAP(3) and MultiSWAP(4) share one entry.
    return type(op) if isinstance(op, Op) and op.is_parametric else op


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

        key = _key(op)
        if not isinstance(key, Op | type(Op)):
            raise TypeError(f"Register the Op family, got {op!r}.")

        if key in self._entries.maps[0] and not replace:
            raise ValueError(
                f"{key.name} is already registered and replace argument is False."
            )

        self._entries[key] = impl
        return impl

    def resolve(self, op: Op | type[Op]) -> Bloq | None:
        impl = self._entries.get(_key(op))
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
        return _key(op) in self._entries


DEFAULT_REPOSITORY = BloqRepository()


def register_op(
    op: Op | type[Op],
    impl: BloqImpl | None = None,
    *,
    repository: BloqRepository = DEFAULT_REPOSITORY,
    replace: bool = False,
):
    return repository.register(op, impl, replace=replace)
