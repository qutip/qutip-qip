import copy
import inspect
import weakref
from dataclasses import fields
from functools import cached_property
from operator import attrgetter
from typing import ClassVar

from qutip import Qobj
from qutip_qip.operations.op import Op, op_factory
from qutip_qip.operations.utils import cached_method
from qutip_qip.utils import convert_type_input_to_sequence
from qutip_qip.typing import Real


class Unitary(Op):
    self_inverse: ClassVar[bool] = False
    is_unitary: ClassVar[bool] = True
    is_clifford: ClassVar[bool] = False
    is_controlled: ClassVar[bool] = False
    config = Op.config | {"self_inverse", "is_clifford", "is_controlled"}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not cls.is_unitary:
            raise TypeError(f"{cls.name} is not unitary, so it has no inverse.")

        if "dtype" not in inspect.signature(cls.compute_qobj).parameters:
            raise TypeError(f"{cls.name}.compute_qobj must accept a dtype parameter.")

        for attr in ("self_inverse", "is_clifford"):
            if not isinstance(inspect.getattr_static(cls, attr), bool | property):
                raise TypeError(f"{cls.name}: {attr} must be a bool.")

        if cls.self_inverse is True and "inverse" in vars(cls):
            raise TypeError(
                f"{cls.name} is self_inverse, so it must not define inverse."
            )

    @cached_method
    def get_qobj(self, dtype: str = "dense") -> Qobj:
        return self.compute_qobj(dtype)

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        """The op's matrix. Any op may define one, from X to QFT(n); the default is to have none."""
        raise NotImplementedError(f"{self.name} has no matrix, decompose it instead.")

    def inverse(self) -> "Unitary | None":
        """Ops with a closed form should overwrite this."""
        return self if self.self_inverse else None

    @property
    def has_matrix(self) -> bool:
        return type(self).compute_qobj is not Unitary.compute_qobj


class AngleUnitary(Unitary):
    """Parametric gate whose parameters are all real angles."""

    def validate_params(self) -> None:
        for v in self.params:
            if isinstance(v, bool) or not isinstance(
                v, Real
            ):  # bool is an int in Python, but not an angle from our perspective
                raise TypeError(f"{self.name}: parameters must be real, got {v!r}.")


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


@op_factory
def get_unitary_gate(name: str, U: Qobj) -> Unitary:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    _name = name
    if not U.isoper or U.dims[0] != U.dims[1]:
        raise ValueError(
            "U must be a square operator with equal input and output dims."
        )

    if not U.isunitary:
        raise ValueError("U must be unitary.")

    class _CustomGate(Unitary):
        __qualname__ = _name
        name = _name
        num_qreg = len(U.dims[0])
        qreg_dim = tuple(U.dims[0])
        self_inverse = U == U.dag()

        def compute_qobj(self, dtype=U.dtype):
            return U.to(dtype)

    return _CustomGate
