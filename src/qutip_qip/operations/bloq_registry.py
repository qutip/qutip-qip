from __future__ import annotations
from collections import ChainMap
from collections.abc import Callable
from dataclasses import dataclass, field

from qutip_qip.operations.bloq import BloqBuilder, Bloq
from qutip_qip.operations.op import Op, ParametricOp

# The implementation of an Op (or ParametricOp) is just a function
# which takes a BloqBuilder instances and adds the relevant gates.
BloqImpl = Callable[..., None]


def _name_of(op: Op | type[ParametricOp]) -> str:
    if isinstance(op, Op) or (isinstance(op, type) and issubclass(op, ParametricOp)):
        return op.name

    if isinstance(op, ParametricOp):
        raise TypeError(
            f"Register the class {type(op).__name__}, not the value {op!r}."
        )
    raise TypeError(f"Expected an Op or a ParametricOp subclass, got {op!r}.")


def _with_params(spec: type[ParametricOp], impl: BloqImpl) -> BloqImpl:
    def bound(op: Op, builder: BloqBuilder) -> None:
        impl(spec(*op.params), builder)

    return bound


@dataclass(eq=False)
class BloqRepository:
    """
    Maps Op names to its implementation. The actual implemented is lazy and only constructed during the runtime.
    """

    # The entries are mapped using string since Op carries params and the
    # registry does not care about them e.g. MultiSWAP(3), MultiSWAP(4) are two different Op values
    # but share the same implementation.

    # TODO: introduce namespace here (only the key changes e.g. std.MultiSWAP)
    _entries: ChainMap[str, BloqImpl] = field(default_factory=ChainMap, init=False)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def register(
        self,
        op: Op | type[ParametricOp],
        impl: BloqImpl | None = None,
        replace: bool = False,
    ):
        if impl is None:
            return lambda f: self.register(op, f, replace=replace)

        name = _name_of(op)
        if name in self._entries.maps[0] and not replace:
            raise ValueError(
                f"{name!r} is already registered and replace argument is False."
            )

        if isinstance(op, type) and issubclass(op, ParametricOp):
            self._entries[name] = _with_params(op, impl)
        else:
            self._entries[name] = impl

        return impl

    def resolve(self, op: Op) -> Bloq | None:
        impl = self._entries.get(op.name)
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


DEFAULT_REPOSITORY = BloqRepository()


def register_op(
    op: Op | type[ParametricOp],
    impl: BloqImpl | None = None,
    *,
    repository: BloqRepository = DEFAULT_REPOSITORY,
    replace: bool = False,
):
    return repository.register(op, impl, replace=replace)
