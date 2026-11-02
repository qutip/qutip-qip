from enum import StrEnum

from qutip_qip.operations.op import Op


class ClassicalControlCheck(StrEnum):
    EQ = "EQ"
    NEQ = "NEQ"
    GT = "GT"
    LT = "LT"
    GTE = "GTE"  # This can be subimplemented using GT - 1
    LTE = "LTE"


class Label(Op):
    """A static marker in the instruction list."""

    uuid: str


class Conditional(Op):
    """Classical conditional control flow statements"""

    num_creg = 1
    label: Label


# TODO: Implement this in future using NOT + Cbnz
# So we won't require 3 classes, only one
class Cbz(Conditional):
    "Conditional branch on zero"


class Cbnz(Conditional):
    "Conditional branch on non-zero i.e. 1."
