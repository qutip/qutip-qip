import copy
import inspect
import weakref
from dataclasses import fields
from functools import cached_property
from operator import attrgetter
from typing import ClassVar

from qutip import Qobj
from qutip_qip.operations.op import op_factory
from qutip_qip.operations.unitary import Modifier, Unitary
from qutip_qip.operations.utils import cached_method


class Inverse(Unitary):
    """
    Inverse() gives the inverse of a value like U or of a family like QFT.
    It is also the base class of every inverse that has no closed form.
    """

    target: ClassVar[Unitary | type[Unitary]]
    _cache: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    def __new__(cls, *args, **kwargs):
        if cls is Inverse:
            return _invert(*args, **kwargs)

        # It must be instantiated with its parameters e.g. Inverse(RX)(0.5) is RX(-0.5) i.e. its the closed form.
        closed = cls.target(*args, **kwargs).inverse() if cls.is_parametric else None
        return super().__new__(cls) if closed is None else closed

    @cached_property
    def target_unitary(self):
        return self.target(*self.params) if self.params else self.target

    @property
    def has_matrix(self) -> bool:
        return self.target_unitary.has_matrix

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return self.target_unitary.get_qobj(dtype).dag()

    @cached_method
    def inverse(self) -> Unitary:
        return self.target_unitary

    def __repr__(self) -> str:
        return f"{self.target_unitary!r}dag"


def _invert(op: Unitary | type[Unitary]) -> Unitary | type[Unitary]:
    """What Inverse(op) returns."""
    family = op if isinstance(op, type) else type(op)

    if issubclass(family, Inverse) and family is not Inverse:
        # Inverse(Inverse(QFT)) is QFT, and the same for a value.
        return op.target if op is family else op.inverse()

    if issubclass(family, Unitary) and family.self_inverse is True:
        return op

    if isinstance(op, Unitary) and not op.params:
        if (closed_form := op.inverse()) is not None:
            return closed_form  # Inverse(T) is Tdag
    return _make_inverse(op)


@op_factory
def _make_inverse(op: Unitary | type[Unitary]) -> Inverse | type[Inverse]:
    is_family = isinstance(op, type)
    family = op if is_family else type(op)

    if not issubclass(family, Unitary) or inspect.isabstract(family):
        raise TypeError(f"Inverse takes a unitary like RX or QFT, got {op!r}.")

    if (cached := Inverse._cache.get(op)) is not None:
        return cached

    def from_target(getter):
        # A config value, such as num_qreg, that the inverse takes from its target.
        # E.g. For QFT, it depends on the parameters so each instance has to look it up.
        if is_family and op.is_derived:
            return property(lambda self: getter(self.target_unitary))
        return getter(op)

    class _Inverse(Inverse):
        # Otherwise every one of these classes is named _Inverse.
        __qualname__ = f"{family.__qualname__}dag"
        target = op

        # The target's parameters, e.g. theta in RX, become the inverse's parameters.
        __annotations__ = {f.name: f.type for f in fields(op)}
        locals().update({f.name: copy.copy(f) for f in fields(op)})
        locals().update(
            {attr: from_target(attrgetter(attr)) for attr in Unitary.config}
        )

        name = from_target(lambda t: f"{t.name}dag")
        latex_str = from_target(lambda t: rf"{t.latex_str}^\dagger")

    Inverse._cache[op] = _Inverse
    return _Inverse
