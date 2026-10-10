from __future__ import annotations

from qutip_qip.operations import Bloq, Op


class BloqRepository:
    def __init__(self, parent: BloqRepository | None = None):
        self._parent = parent
        self._entries: dict[str, list[BloqEntry]] = {}


DEFAULT_REPOSITORY = BloqRepository()


def resolve_op(op: Op, repository: BloqRepository = DEFAULT_REPOSITORY) -> Bloq | None:
    pass


def compile_bloq(bloq: Bloq, primitives: tuple[Bloq, ...]) -> Bloq:
    pass
