import copyreg
import inspect
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field, fields
from functools import partial, wraps
from typing import ClassVar, dataclass_transform, get_origin

from qutip_qip.typing import Int
from qutip_qip.utils import convert_type_input_to_sequence


def _check_config(op) -> None:
    """
    Validate an Op. We need to do it at two places due to non-derived, derived.

    e.g. class Toffoli(Op): num_qreg = 3, is known when the class is created.
    Checking it in __init_subclass__ makes an authoring error fail at import and it is the only chance for non-parametric op.
    """

    if not isinstance(op.num_qreg, Int) or op.num_qreg < 0:
        raise ValueError(f"{op.name}: num_qreg must be an integer >= 0.")

    if not isinstance(op.num_creg, Int) or op.num_creg < 0:
        raise ValueError(f"{op.name}: num_creg must be an integer >= 0.")

    if (
        not isinstance(op.qreg_dim, tuple)
        or len(op.qreg_dim) != op.num_qreg
        or any(not isinstance(d, Int) or d < 2 for d in op.qreg_dim)
    ):
        raise ValueError(
            f"{op.name}: qreg_dim={op.qreg_dim} needs {op.num_qreg} entries, each >= 2."
        )

    if not isinstance(op.is_unitary, bool):
        raise TypeError("is_unitary must be of type bool")


class _OpMeta(ABCMeta):

    # Instead of asking the user to initialize X() or custom Op so that we support isinstance throughout
    # We instead do it directly in the Metaclass (where the class gets constructed)
    # This also ensure non-parametric Ops like X stay a singleton (no duplicates), also saves memory
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns, **kw)
        cls._sealed = True  # from here on class is read-only

        if cls.__abstractmethods__ or cls.is_parametric:
            return cls
        cls._value = cls()
        return cls._value

    def __call__(cls, *args, **kwargs):
        if "_value" in vars(cls):
            raise TypeError(
                f"{cls.__qualname__} is a value, not a parametric op, use it directly."
            )
        return super().__call__(*args, **kwargs)

    def __setattr__(cls, name, value):
        if not name.startswith("_") and "_sealed" in vars(cls):
            raise AttributeError(f"{cls.__name__}.{name} is read-only.")
        super().__setattr__(name, value)


# Think of userdefined Op as Op Factory, takes params and returns a Op
# We need it because we want something like
# class MultiSWAP(Op):
#   n: int  # this is a user defined parameter
#
#   def validate_params(self): ...

# If we don't have this dataclass, the Parametric Ops will have to
# define the parameters in __init__ (Pythonic but slightly less readable)


# Note: dataclass transform below is just a typehint to indicate that
# the inherited classes will be a dataclass.
@dataclass_transform(frozen_default=True)
class Op(ABC, metaclass=_OpMeta):
    """
    Base class of every operation. Operations are defined by subclassing.
    """

    # configuration
    name: ClassVar[str]
    num_creg: ClassVar[int] = 0
    qreg_dim: ClassVar[tuple[int, ...]] = ()
    latex_str: ClassVar[str]
    is_unitary: ClassVar[bool] = False

    @property
    @abstractmethod
    def num_qreg(self) -> int: ...

    # derived from configuration
    is_parametric: ClassVar[bool] = False
    is_derived: ClassVar[bool] = False
    config: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "num_qreg",
            "num_creg",
            "qreg_dim",
            "latex_str",
            "is_unitary",
        }
    )

    def __mro_entries__(self, bases):
        return (type(self),)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # Each class adds its own config names; the full set is the union over its bases.
        cls.config = frozenset().union(
            *(vars(base).get("config", ()) for base in cls.__mro__)
        )

        # Say user writes num_qreg: int in an inherited dataclass and this will generate __init__(self, num_qreg=3).
        # Now num_qreg is a parameter i.e it shows in __eq__ and Op(num_qreg=7) is accepted (which we don't want).
        # So we keep the annotation for field matching _CONFIG just we can ignore them by making them a ClassVar.
        ann = dict(cls.__annotations__)
        for attr in cls.config & ann.keys():
            if get_origin(ann[attr]) is not ClassVar:
                ann[attr] = ClassVar[ann[attr]]
        cls.__annotations__ = ann

        #  If name attribute in subclass is not defined, set it to the name of the subclass
        if "name" not in vars(cls):
            cls.name = cls.__name__
        if "latex_str" not in vars(cls):
            cls.latex_str = cls.name

        dataclass(frozen=True, repr=False, eq=False)(cls)

        # If a user defines a keyword "only" field (kw) in a dataclass, we reject it
        if any(f.kw_only or not f.init for f in fields(cls)):
            raise TypeError(f"{cls.__name__}: parameters must be positional fields.")

        cls.is_parametric = bool(fields(cls))

        if isinstance(vars(cls).get("num_qreg"), Int) and "qreg_dim" not in vars(cls):
            cls.qreg_dim = (2,) * cls.num_qreg

        cls.is_derived = any(
            isinstance(inspect.getattr_static(cls, attr, None), property)
            for attr in cls.config
        )

        # Skip abstract classes, below checks don't apply to abstract classes
        if inspect.isabstract(cls):
            return

        if not cls.is_derived:
            _check_config(cls)

        # Just for safety, every derived class must be parametric
        if cls.is_derived and not cls.is_parametric:
            raise TypeError(
                f"{cls.__name__}: config can only be derived from parameters, add a new parameter."
            )

    def __post_init__(self) -> None:
        if self.is_derived:
            if not self.qreg_dim:
                # The dataclass is frozen
                object.__setattr__(self, "qreg_dim", (2,) * self.num_qreg)

            _check_config(self)
        self.validate_params()

    def validate_params(self) -> None:
        """Overwritten for Parametric Ops. Raise ValueError on invalid parameters."""

    @property
    def params(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    @property
    def arg_value(self) -> tuple:  # deprecated alias for the 72 call sites
        warnings.warn(
            "arg_value argument for parametric gates is deprecated",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.params

    def __repr__(self) -> str:  # e.g. "MultiSWAP(3)" for drawing
        if not self.params:
            return self.name
        return f"{self.name}({', '.join(map(str, self.params))})"

    @property
    def _key(self):
        return (type(self), self.params)

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other) -> bool:
        return isinstance(other, Op) and self._key == other._key

    def __reduce__(self):
        cls = type(self)
        if self.params:
            return cls, self.params
        return vars(cls).get("_factory") or cls.__qualname__

    # Ops are immutable, so a copy is the value itself
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    # TODO: For dev backward compatibility only, remove later
    @property
    def num_qubits(self) -> int:
        return self.num_qreg


# Class to keep track of this Op on this qubit, cbit
@dataclass(frozen=True, slots=True)
class OpInstruction:
    op: Op
    qreg: tuple[int, ...] = tuple()
    creg: tuple[int, ...] = tuple()
    style: dict = field(default_factory=dict)  # For circuit draw

    def __post_init__(self):
        if not isinstance(self.op, Op):
            raise TypeError(f"op must be an operation value, got {self.op!r}.")

        if len(self.qreg) != self.op.num_qreg:
            raise ValueError(
                f"{self.op.name} acts on {self.op.num_qreg} qubits, got {len(self.qreg)}."
            )

        if len(self.creg) != self.op.num_creg:
            raise ValueError(
                f"{self.op.name} uses {self.op.num_creg} cbits, got {len(self.creg)}."
            )

    def __repr__(self) -> str:
        return f"OpInstruction({repr(self.op)}, qreg={self.qreg}, creg={self.creg})"


# In future if we want to pickle Ops or use the multiprocessing package
# the Ops created via functions like get_unitary_gate, controlled won't
# reduce properly due to the import path issue. That's why we have to define op_factory.
copyreg.pickle(_OpMeta, lambda cls: vars(cls).get("_factory") or cls.__qualname__)


def op_factory(build):
    @wraps(build)
    def factory(*args, **kwargs):
        # A value's parameters carry over: controlled(RX(0.5)) is controlled(RX)(0.5).
        params = args[0].params if args and isinstance(args[0], Op) else ()
        if params:
            args = (type(args[0]), *args[1:])
        made = build(*args, **kwargs)

        # A Op factory-made class (e.g. via controlled, get_unitary_gate etc.) has no import path, so it pickles as the call that made it.
        family = made if made.is_parametric else type(made)
        family._factory = (partial(factory, *args, **kwargs), ())
        return made(*params) if params else made

    return factory
