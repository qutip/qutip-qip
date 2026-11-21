import copy
import inspect
import weakref
from dataclasses import fields
from operator import attrgetter
from typing import ClassVar

from qutip import Qobj
from qutip_qip.operations.op import op_factory
from qutip_qip.operations.unitary import Modifier, Unitary


class Inverse(Modifier):
    """
    Inverse() gives the inverse of a value like U or of a family like QFT.
    It is also the base class of every inverse that has no closed form.
    """

    _cache: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    def __new__(cls, *args, **kwargs):
        if cls is Inverse:
            return _invert(*args, **kwargs)

        self = super().__new__(cls)
        if cls.is_parametric:
            target = cls.target(*args, **kwargs)
            if (closed_form := target.inverse()) is not None:
                return closed_form  # Inverse(RX)(0.5) is RX(-0.5)
            object.__setattr__(self, "target", target)  # the dataclass is frozen
        return self

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return self.target.get_qobj(dtype).dag()

    def inverse(self) -> Unitary:
        return self.target

    def __repr__(self) -> str:
        return f"{self.target!r}dag"


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
            return property(lambda self: getter(self.target))
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
