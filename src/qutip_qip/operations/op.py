from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from uuid import uuid4

from qutip_qip.typing import Int
from qutip_qip.utils import convert_type_input_to_sequence

# TODO: Add typing bound constraint to this
P = TypeVar("P")


# @dataclass(frozen=True)  # TODO: Add slots when minimum Python is bumped to 3.14
@dataclass
class Op:
    _name: str
    uuid: str = field(default_factory=lambda: uuid4().hex)
    params: tuple[P, ...] = ()

    @property
    def name(self) -> str:  # e.g. "MultiSWAP(3)" for drawing
        if not self.params:
            return self._name
        else:
            return f"{self._name}({', '.join(map(str, self.params))})"

    qreg_dim: tuple[int, ...] = ()
    num_creg: int = 0

    @property
    def num_qreg(self) -> int:
        return len(self.qreg_dim)

    unitary: bool = True
    self_inverse: bool = False


# Think of Parametric Op as Op Factory, takes params and returns a Op
# TODO: Update this typing when updating min Python version to 3.12
class ParametricOp(ABC, Generic[P]):
    def __call__(self, *params) -> Op: ...
    def validate_params(self, *params: P) -> None: ...


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
