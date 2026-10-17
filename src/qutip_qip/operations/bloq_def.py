import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar, dataclass_transform

from qutip_qip.operations.bloq import BloqBuilder

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

# Note: dataclass transform is just a typehint to indicate that
# the inherited classes will be a dataclass.

# We need it because we want something like
# class MultiSWAPDef(BloqDef):
#   n: int  # this is a user defined parameter
#
#   def build(self, bloq): ...

# If we don't have this dataclass, the Bloqs for Parametric Ops will have to
# define the parameters in __init__ (Pythonic but slightly less readable)


@dataclass_transform(frozen_default=True)
class BloqDef(ABC):
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
        # Now num_qreg is a parameter i.e it shows in __eq__ and MajorityDef(num_qreg=7) is accepted (which we don't want).
        # num_qreg is configuration, not a parameter of the Op

        # So we keep the annotation for field matching _RESERVED_ATTRS just we can ignore them by making them a ClassVar.
        ann = dict(cls.__dict__.get("__annotations__", {}))
        for attr in _RESERVED_ATTRS & ann.keys():
            ann[attr] = ClassVar[ann[attr]]

        cls.__annotations__ = ann
        dataclasses.dataclass(frozen=True)(cls)

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
        return tuple(getattr(self, f.name) for f in dataclasses.fields(self))

    @property
    def num_qreg(self) -> int:
        if type(self).qreg_dim is not BloqDef.qreg_dim:  # user overrode qreg_dim
            return len(self.qreg_dim)
        raise TypeError(f"{type(self).__name__} must define num_qreg or qreg_dim")

    @property
    def qreg_dim(self) -> tuple[int, ...]:
        return (2,) * self.num_qreg

    @abstractmethod
    def build(self, bloq: BloqBuilder) -> None:
        raise NotImplementedError
