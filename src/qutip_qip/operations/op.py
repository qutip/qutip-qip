import copyreg
import inspect
from abc import ABC, ABCMeta
from dataclasses import dataclass, field, fields
from functools import partial, wraps
from typing import ClassVar, dataclass_transform, get_origin

from qutip_qip.operations.utils import class_or_instance_method
from qutip_qip.typing import Int
from qutip_qip.utils import convert_type_input_to_sequence


def _check_config(op) -> None:
    """
    Validate an Op. We need to do it at two places due to non-derived, derived.

    e.g. class Toffoli(Op): num_qreg = 3, is known when the class is created.
    Checking it in __init_subclass__ makes an authoring error fail at import and it is the only chance for non-parametric op.

    e.g. MCX with num_qreg as a property over n, cannot be checked until there is an instance with a value for n. So __post_init__ checks it there
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

    if not isinstance(op.unitary, bool):
        raise TypeError("unitary must be of type bool")

    if not isinstance(op.self_inverse, bool):
        raise TypeError("self_inverse must be of type bool")


class OpMeta(ABCMeta):
    """_key if present is a custom key to be used to hash"""

    # To avoid checking the base class for hierarchy.
    # Otherwise having a intermediatery class e.g. _TwoQubitControlledGate for CX
    # makes CX == controlled(X) check fail.
    # For further details, read about ABCMeta cache
    __instancecheck__ = type.__instancecheck__
    __subclasscheck__ = type.__subclasscheck__

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._sealed = True  # class creation, including __init_subclass__ is complete

    # We require this so that class attributes are not overwritable e.g. X.num_qreg fails
    # frozen=True in dataclass generated for each Op works only for instance (i.e. Parametric Op)
    def __setattr__(cls, name, value):
        if not name.startswith("_") and "_sealed" in cls.__dict__:
            raise AttributeError(f"{cls.__name__}.{name} is read-only.")
        super().__setattr__(name, value)

    def __eq__(cls, other):
        key = vars(cls).get("_key")
        return cls is other or (
            key is not None
            and isinstance(other, OpMeta)
            and key == vars(other).get("_key")
        )

    def __hash__(cls):
        key = vars(cls).get("_key")
        return type.__hash__(cls) if key is None else hash(key)


# Think of userdefined Op as Op Factory, takes params and returns a Op
# We need it because we want something like
# class MultiSWAP(Op):
#   n: int  # this is a user defined parameter
#
#   def validate_params(self): ...

# If we don't have this dataclass, the Parametric Ops will have to
# define the parameters in __init__ (Pythonic but slightly less readable)


# Note: dataclass transform is just a typehint to indicate that
# the inherited classes will be a dataclass.
@dataclass_transform(frozen_default=True)
class Op(ABC, metaclass=OpMeta):
    """
    Base class of every operation. Operations are defined by subclassing.
    """

    # configuration
    name: ClassVar[str]
    num_qreg: ClassVar[int] = 0
    num_creg: ClassVar[int] = 0
    qreg_dim: ClassVar[tuple[int, ...]] = ()
    self_inverse: ClassVar[bool] = False
    unitary: ClassVar[bool] = False

    # derived from configuration
    is_controlled: ClassVar[bool] = False
    is_parametric: ClassVar[bool] = False
    is_derived: ClassVar[bool] = False  #
    config: ClassVar[frozenset[str]] = frozenset(
        {"name", "num_qreg", "num_creg", "qreg_dim", "unitary", "self_inverse"}
    )

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError(f"{type(self).__name__} class can't be instantiated.")

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # Each class adds its own config names; the full set is the union over its bases.
        cls.config = frozenset().union(
            *(vars(base).get("config", ()) for base in cls.__mro__)
        )

        # Example:
        # Say user writes num_qreg: int in the child dataclass and this will generate __init__(self, num_qreg=3).
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

        dataclass(frozen=True, repr=False, eq=False)(cls)

        # If a user defines a keyword "only" field in a dataclass, we reject it
        if any(f.kw_only or not f.init for f in fields(cls)):
            raise TypeError(f"{cls.__name__}: parameters must be positional fields.")

        cls.is_parametric = bool(fields(cls))
        if cls.is_parametric:
            cls.params = Op.params
        else:
            cls.params = ()
            cls.__init__ = Op.__init__

        # Check the config (num_qreg, num_creg etc.) if none of them are parametric
        cls.is_derived = any(
            isinstance(inspect.getattr_static(cls, attr, None), property)
            for attr in cls.config
        )
        if not cls.is_derived:
            if "num_qreg" in vars(cls) and "qreg_dim" not in vars(cls):
                cls.qreg_dim = (2,) * cls.num_qreg
            _check_config(cls)

        if not inspect.isabstract(cls) and cls.is_derived and not cls.is_parametric:
            raise TypeError(
                f"{cls.__name__}: config can only be derived from parameters, add a new parameter."
            )

    def __post_init__(self) -> None:  # parametric instances only
        if self.is_derived:
            if not self.qreg_dim:
                object.__setattr__(
                    self, "qreg_dim", (2,) * self.num_qreg
                )  # The dataclass is frozen
            _check_config(self)

        self.validate_params()

    def validate_params(self) -> None:
        """Overwritten for Parametric Ops. Raise ValueError on invalid parameters."""

    @property
    def params(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    @class_or_instance_method
    def inverse(cls_ins):
        # To disallow something like RX.inverse()
        if not is_op(cls_ins):
            raise TypeError(
                f"{cls_ins.name} is parametric, build an instance first e.g. {cls_ins.name}(...).inverse()."
            )

        if not cls_ins.unitary:
            raise TypeError(f"{cls_ins.name} is not unitary, so it has no inverse.")

        if cls_ins.self_inverse:
            return cls_ins

        # To avoid circular import
        from qutip_qip.operations.inverse import inverse

        if isinstance(cls_ins, type):
            return inverse(cls_ins)
        return inverse(type(cls_ins))(*cls_ins.params)

    def __repr__(self) -> str:  # e.g. "MultiSWAP(3)" for drawing
        if not self.params:
            return self.name
        return f"{self.name}({', '.join(map(str, self.params))})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Op)
            and type(self) == type(other)  # noqa: E721
            and self.params == other.params
        )

    def __hash__(self) -> int:
        return hash((type(self), self.params))


# Class to keep track of this Op on this qubit, cbit
@dataclass(frozen=True, slots=True)
class OpInstruction:
    op: Op
    qreg: tuple[int, ...] = tuple()
    creg: tuple[int, ...] = tuple()
    style: dict = field(default_factory=dict)  # For circuit draw

    def __post_init__(self):
        if not is_op(self.op):
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
        label = self.op.name if isinstance(self.op, type) else repr(self.op)
        return f"OpInstruction({label}, qreg={self.qreg}, creg={self.creg})"


# In future if we want to pickle Ops or use the multiprocessing package
# the Ops created via functions like get_unitary_gate, controlled won't
# reduce properly due to the import path issue. That's why we have to define op_factory.
copyreg.pickle(OpMeta, lambda cls: vars(cls).get("_factory") or cls.__qualname__)


def op_factory(build):
    @wraps(build)
    def factory(*args, **kwargs):
        if args and isinstance(args[0], Op):
            op, *rest = args
            return factory(type(op), *rest, **kwargs)(*op.params)
        cls = build(*args, **kwargs)
        cls._factory = (partial(factory, *args, **kwargs), ())
        return cls

    return factory


# Helper function for other modules
def is_op(obj, kind: type[Op] = Op) -> bool:
    if isinstance(obj, kind):
        return True

    return (
        isinstance(obj, type)
        and issubclass(obj, kind)
        and obj is not Op
        and not obj.is_parametric
        and not inspect.isabstract(obj)
    )
