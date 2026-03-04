# annotations can be removed when base version is Python 3.14
from __future__ import annotations
from functools import cached_property
from dataclasses import dataclass

__all__ = ["NameSpace", "NameSpaceRegistry", "STD_NS"]


class _SingletonMeta(type):
    """
    Note this is not a thread-safe implementation of Singleton.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class _GlobalRegistry(metaclass=_SingletonMeta):
    def __init__(self):
        self._registry: dict[NameSpace, dict[str, any]] = {}

    def register(self, namespace: NameSpace, name: str, item: any) -> None:
        """Safely adds an item to the specific namespace."""
        if namespace not in self._registry:
            self._registry[namespace] = {}

        name_upper = name.upper()
        if name_upper in self._registry[namespace]:
            raise ValueError(
                f"'{name_upper}' already exists in namespace '{namespace}'"
            )

        self._registry[namespace][name_upper] = item

    def get(self, namespace: NameSpace, name: str) -> any:
        """Retrieves an item from a specific namespace."""
        try:
            return self._registry[namespace][name.upper()]
        except KeyError:
            raise KeyError(
                f"'{name.upper()}' not found in namespace '{namespace}'."
            )


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

    @cached_property
    def name(self) -> str:
        if self.parent:
            return f"{self.parent.name}.{self.local_name}"
        return self.local_name

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return str(self)


NameSpaceRegistry = _GlobalRegistry()
STD_NS = NameSpace("std")  # DEFAULT NAMESPACE
