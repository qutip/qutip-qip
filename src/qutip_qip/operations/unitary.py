import copy
import inspect
import weakref
from abc import abstractmethod
from dataclasses import fields
from functools import cached_property
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

    _made: ClassVar[weakref.WeakValueDictionary] = weakref.WeakValueDictionary()

    _derived = {
        attr: lambda inv, t, attr=attr: getattr(t, attr) for attr in Unitary.config
    } | {
        "name": lambda inv, t: f"{t.name}dag",
        "latex_str": lambda inv, t: rf"{t.latex_str}^\dagger",
    }

    @property
    @abstractmethod
    def target(self) -> Unitary | type[Unitary]:
        """The op being inverted: a value like U or a family like QFT."""

    def __init_subclass__(cls) -> None:
        per_instance = isinstance(cls.target, type) and cls.target.is_derived
        for attr, rule in cls._derived.items():
            setattr(
                cls,
                attr,
                (
                    property(lambda self, rule=rule: rule(self, self.target_unitary))
                    if per_instance
                    else rule(cls, cls.target)
                ),
            )

        if isinstance(cls.target, type):
            # The target's parameters, e.g. theta in RX, become cls's parameters.
            cls.__annotations__ = {f.name: f.type for f in fields(cls.target)} | dict(
                cls.__annotations__
            )
            for f in fields(cls.target):
                setattr(cls, f.name, copy.copy(f))

        super().__init_subclass__()

    def __new__(cls, *args, **kwargs):
        if cls is Inverse:
            return _inverse(*args, **kwargs)

        # A family called with its parameters e.g. Inverse(RX)(0.5) is RX(-0.5) i.e. its the closed form.
        closed = cls.target(*args, **kwargs).inverse() if cls.is_parametric else None
        return super().__new__(cls) if closed is None else closed

    @cached_property
    def target_unitary(self) -> Unitary:
        return self.target(*self.params) if self.params else self.target

    @property
    def has_matrix(self) -> bool:
        return self.target_unitary.has_matrix

    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        return self.target_unitary.get_qobj(dtype).dag()

    def inverse(self) -> Unitary:
        return self.target_unitary

    def __repr__(self) -> str:
        return f"{self.target_unitary!r}dag"


def _inverse(op: Unitary | type[Unitary]) -> Unitary | type[Unitary]:
    """What Inverse(op) returns."""
    if isinstance(op, Unitary) and (closed := op.inverse()) is not None:
        return closed  # Inverse(T) is Tdag
    if isinstance(op, type) and issubclass(op, Inverse) and op is not Inverse:
        return op.target  # Inverse(Inverse(QFT)) is QFT
    return _wrap(op)


@op_factory
def _wrap(op):
    family = op if isinstance(op, type) else type(op)
    if not issubclass(family, Unitary) or inspect.isabstract(family):
        raise TypeError(f"Inverse takes a unitary like RX or QFT, got {op!r}.")

    if (made := Inverse._made.get(op)) is None:

        class _Inverse(Inverse):
            __qualname__ = (
                f"{family.__qualname__}dag"  # otherwise every one is named _Inverse
            )
            target = op

        made = Inverse._made[op] = _Inverse
    return made


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
