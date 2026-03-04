from __future__ import annotations  # To be removed base version is Python 3.14
from functools import cached_property
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NameSpace:
    local_name: str
    parent: NameSpace | None = None

    def __post_init__(self):
        if "." in self.local_name:
            raise ValueError(
                f"Namespace local_name '{self.local_name}' cannot contain dots. "
                f"Dots are reserved for hierarchical resolution."
            )
        if not self.local_name.isidentifier():
            raise ValueError(
                f"'{self.local_name}' is not a valid namespace identifier."
            )

    @property
    @cached_property
    def name(self) -> str:
        if self.parent:
            return f"{self.parent.name}.{self.local_name}"
        return self.local_name

    def __str__(self) -> str:
        return self.name


# The Root
DEFAULT = NameSpace("__default__")

# For standard gates and measurement
STD = NameSpace("std")
