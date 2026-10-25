from dataclasses import dataclass, field, fields
from typing import ClassVar, dataclass_transform, TypeVar

from qutip_qip.typing import Int
from qutip_qip.utils import convert_type_input_to_sequence

# TODO: Add typing bound constraint to this
# TODO: Update this typing when updating min Python version to 3.12
P = TypeVar("P")

_RESERVED_ATTRS = frozenset(
    {
        "name",
        "unitary",
        "self_inverse",
        "num_qreg",
        "num_creg",
        "qreg_dim",
    }
)


# TODO: Add slots when minimum Python is bumped to 3.14
@dataclass(frozen=True)
class Op:
    _name: str

    @property
    def name(self) -> str:  # e.g. "MultiSWAP(3)" for drawing
        if not self.params:
            return self._name
        else:
            return f"{self._name}({', '.join(map(str, self.params))})"

    @property
    def num_qreg(self) -> int:
        return len(self.qreg_dim)

    num_creg: int = 0
    params: tuple[P, ...] = ()
    unitary: bool = True
    self_inverse: bool = False
    qreg_dim: tuple[int, ...] = ()


# Think of Parametric Op as Op Factory, takes params and returns a Op
# We need it because we want something like
# class MultiSWAP(ParametricOp):
#   n: int  # this is a user defined parameter
#
#   def validate_params(self): ...

# If we don't have this dataclass, the Parametric Ops will have to
# define the parameters in __init__ (Pythonic but slightly less readable)

# Note: dataclass transform is just a typehint to indicate that
# the inherited classes will be a dataclass.


@dataclass_transform(frozen_default=True)
class ParametricOp:
    name: ClassVar[str]  # defaults to class name minus "Def"
    unitary: ClassVar[bool] = True
    self_inverse: ClassVar[bool] = False
    num_creg: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "name" not in vars(cls):
            cls.name = cls.__name__.removesuffix("Def")

        # Example:
        # Say user writes num_qreg: int in the child dataclass and this will generate __init__(self, num_qreg=3).
        # Now num_qreg is a parameter i.e it shows in __eq__ and ParametricOp(num_qreg=7) is accepted (which we don't want).
        # num_qreg is configuration, not a parameter of the Op

        # So we keep the annotation for field matching _RESERVED_ATTRS just we can ignore them by making them a ClassVar.
        ann = dict(cls.__dict__.get("__annotations__", {}))
        for attr in _RESERVED_ATTRS & ann.keys():
            ann[attr] = ClassVar[ann[attr]]

        cls.__annotations__ = ann
        dataclass(frozen=True)(cls)

    # This function will only be executed in the inherited dataclass.
    def __post_init__(self) -> None:
        self.validate_params()

    def validate_params(self) -> None:
        """
        This can be overwridden in the Parametric child classes to check for user input.
        Raise a ValueError in case of wrong parameter being passed.
        """
        pass

    @property
    def params(self) -> tuple:
        """Returns a tuple of names of parameters."""
        return tuple(getattr(self, f.name) for f in fields(self))

    @property
    def num_qreg(self) -> int:
        if type(self).qreg_dim is not ParametricOp.qreg_dim:  # user overwrote qreg_dim
            return len(self.qreg_dim)
        raise TypeError(f"{type(self).__name__} must define num_qreg or qreg_dim")

    @property
    def qreg_dim(self) -> tuple[int, ...]:
        return (2,) * self.num_qreg  # Unless user overwrote it

    def to_op(self) -> Op:
        return Op(
            self.name,
            qreg_dim=tuple(self.qreg_dim),
            num_creg=self.num_creg,
            unitary=self.unitary,
            self_inverse=self.self_inverse,
            params=self.params,
        )


# Class to keep track of this Op on this qubit, cbit
@dataclass(frozen=True, slots=True)
class OpInstruction:
    op: Op
    qreg: tuple[int, ...] = tuple()
    creg: tuple[int, ...] = tuple()
    style: dict = field(default_factory=dict)  # For circuit draw

    def __post_init__(self):
        if not (
            isinstance(self.op, Op)
            or ((isinstance(self.op, type) and issubclass(self.op, Op)))
        ):
            raise TypeError("op must be a subclass or instance of type Op")

        convert_type_input_to_sequence(Int, "qreg", self.qreg)
        convert_type_input_to_sequence(Int, "creg", self.creg)
