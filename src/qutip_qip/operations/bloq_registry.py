from __future__ import annotations
from collections import ChainMap
from dataclasses import dataclass, field

from qutip_qip.operations import BloqDef, Bloq, Op


@dataclass(eq=False)
class BloqRepository:
    """ """

    # The entries are mapped using string since Op carries params and the
    # registry does not care about them i.e. MultiSWAP(3), MultiSWAP(4) are two different Op values
    # but share the same implementation.

    # TODO: introduce namespace here (only the key changes e.g. std.MultiSWAP)
    _entries: ChainMap[str, type[BloqDef]] = field(default_factory=ChainMap, init=False)

    def register():
        pass

    def resolve():
        pass

    def scope():
        pass


DEFAULT_REPOSITORY = BloqRepository()


def compile_bloq(bloq: Bloq, primitives: tuple[Bloq, ...]) -> Bloq:
    pass
