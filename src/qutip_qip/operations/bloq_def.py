from abc import ABC, abstractmethod
from typing import ClassVar
import inspect

from qutip_qip.operations.bloq import _BloqBuilder, Bloq
from qutip_qip.operations.op import Op


def define_op(bloq_def: BloqDef) -> tuple[Op, Bloq]:
    pass


class BloqDef(_BloqBuilder, ABC):
    name: str  # defaults to class name minus "Def"
    unitary: bool = True
    self_inverse: bool = False
    _fields: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        if "name" not in vars(cls):
            cls.name = cls.__name__.removesuffix("Def")

    # In case the user wants to define a qudit circuit, they can override this
    # e.g. qreg_dim = (3, 3)   # two qutrits
    @property
    def qreg_dim(self) -> tuple[int, ...]:
        return (2,) * self.num_qreg

    @abstractmethod
    def bloq(self) -> None:
        raise NotImplementedError

    def __init__(self) -> None:
        raise TypeError(
            "Inherit from ParametricBloqDef in case you want your Bloq to be parmaetric"
        )


class ParametricBloqDef(BloqDef):
    def __init__(self, *params):
        self.params = params
