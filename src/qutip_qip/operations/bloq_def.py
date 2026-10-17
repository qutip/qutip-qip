from abc import ABC, abstractmethod

from qutip_qip.operations.bloq import BloqBuilder, Bloq


class BloqDef(ABC):
    name: str  # defaults to class name minus "Def"
    unitary: bool = True
    self_inverse: bool = False
    num_creg: int = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "name" not in vars(cls):
            cls.name = cls.__name__.removesuffix("Def")

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

    def compile(self) -> Bloq:
        qreg_dim = tuple(self.qreg_dim)
        builder = BloqBuilder(
            num_qreg=len(qreg_dim),
            num_creg=self.num_creg,
            qreg_dim=qreg_dim,
        )
        self.build(builder)
        return builder.build()
