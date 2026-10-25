from __future__ import annotations
from collections import ChainMap
from dataclasses import dataclass, field

from qutip_qip.operations import BloqBuilder, BloqDef, Bloq, Op


@dataclass(eq=False)
class BloqRepository:
    """
    Maps Op names to the BloqDef for implementing them.
    The actual implemented is lazy and only constructed during the run-time.
    """

    # The entries are mapped using string since Op carries params and the
    # registry does not care about them e.g. MultiSWAP(3), MultiSWAP(4) are two different Op values
    # but share the same implementation.

    # TODO: introduce namespace here (only the key changes e.g. std.MultiSWAP)
    _entries: ChainMap[str, type[BloqDef]] = field(default_factory=ChainMap, init=False)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def register(self, bloqdef: type[BloqDef], op: Op | None = None):
        if not (isinstance(bloqdef, type) and issubclass(bloqdef, BloqDef)):
            raise TypeError(f"Expected a BloqDef subclass, got {bloqdef!r}")

        if bloqdef.name in self._entries.maps[0]:
            raise ValueError(f"{bloqdef.name!r} is already registered.")

        if op is not None:
            self._entries[op._name] = bloqdef
        else:
            self._entries[bloqdef.name] = bloqdef

    def resolve(self, op: Op) -> Bloq | None:
        bloqdef = self._entries.get(op)
        if bloqdef is None:
            return None

        # Constructing the bloq for the given params
        builder = BloqBuilder(op.num_qreg, op.num_creg, op.qreg_dim)
        bloqdef(*op.params).build(builder)
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


def compile_bloq(bloq: Bloq, primitives: tuple[Op, ...]) -> Bloq:
    pass
