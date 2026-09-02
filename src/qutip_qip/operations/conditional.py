from dataclasses import dataclass
from enum import StrEnum

from qutip_qip.operations import Op


class ClassicalControlCheck(StrEnum):
    EQ = "EQ"
    NEQ = "NEQ"
    GT = "GT"
    LT = "LT"
    GTE = "GTE"  # This can be subimplemented using GT - 1
    LTE = "LTE"


class Label(Op):
    """A static marker in the instruction list."""

    def __init__(self, name):
        super(Label, self).__init__(_name=name)


# @dataclass
class Conditional(Op):
    """Classical conditional control flow statements"""

    label: str

    def __init__(self, label: str):
        # Don't make it super(), it will throw an error because slots=True
        # destroys __class__ reference to the original class until Python 3.13
        # Check CPython Issue #90562, TODO: this has been resolved in Python 3.14
        super(Conditional, self).__init__(_name=type(self).__name__)  #
        object.__setattr__(self, "label", label)


# TODO: Implement this in future using NOT + Cbnz
# So we won't require 3 classes, only one
class Cbz(Conditional):
    "Conditional branch on zero"


class Cbnz(Conditional):
    "Conditional branch on non-zero i.e. 1."
