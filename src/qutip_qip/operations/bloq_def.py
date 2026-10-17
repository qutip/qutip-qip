from abc import ABC, abstractmethod

from qutip_qip.operations.bloq import BloqBuilder


class BloqDef(ABC):
    name: str  # defaults to class name minus "Def"
    unitary: bool = True
    self_inverse: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "name" not in vars(cls):
            cls.name = cls.__name__.removesuffix("Def")

    @abstractmethod
    def build(self, bloq: BloqBuilder) -> None:
        raise NotImplementedError
