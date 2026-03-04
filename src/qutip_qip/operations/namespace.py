# annotations can be removed when base version is Python 3.14 (PEP 749)
from __future__ import annotations
from functools import cached_property
from dataclasses import dataclass, field


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
        self._registry: set[NameSpace] = set()

    def register_namespace(self, namespace: NameSpace) -> None:
        """Safely adds an item to the specific namespace."""
        if namespace in self._registry:
            raise ValueError(f"Existing namespace {namespace}")
        self._registry.add(namespace)


GlobalNameSpaceRegistry = _GlobalRegistry()


@dataclass
class NameSpace:
    local_name: str
    parent: NameSpace | None = None
    _registry: dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        if "." in self.local_name:
            raise ValueError(
                f"Namespace local_name '{self.local_name}' cannot contain dots. "
                f"Dots are reserved for hierarchical resolution."
            )
        GlobalNameSpaceRegistry.register_namespace(self)

    @cached_property
    def name(self) -> str:
        if self.parent:
            return f"{self.parent.name}.{self.local_name}"
        return self.local_name

    def register(self, name: str, item: any) -> None:
        """Safely adds an item to the specific namespace."""
        name_upper = name.upper()
        if name_upper in self._registry:
            existing_item = self._registry[name_upper]

            # Tolerate harmless reloads from pytest double-imports or Jupyter cell re-runs
            if getattr(existing_item, "__module__", None) == getattr(
                item, "__module__", None
            ) and getattr(existing_item, "__name__", None) == getattr(
                item, "__name__", None
            ):

                # Silently overwrite with the fresh reloaded class and continue
                self._registry[name_upper] = item
                return

            raise ValueError(
                f"'{name_upper}' already exists in namespace '{self.name}'"
            )

        self._registry[name_upper] = item

    def get(self, name: str) -> any:
        """Retrieves an item from a specific namespace."""
        try:
            return self._registry[name.upper()]
        except KeyError:
            raise KeyError(
                f"'{name.upper()}' not found in namespace '{self.name}'."
            )

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return str(self)


NS_STD = NameSpace("std")  # DEFAULT NAMESPACE
NS_GATE = NameSpace("gates", parent=NS_STD)  # Default Gate Namespace
NS_USER = NameSpace("user")  # Default for anything defined by the user
