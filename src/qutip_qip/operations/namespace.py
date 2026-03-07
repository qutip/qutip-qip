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


class _GlobalNameSpaceRegistry(metaclass=_SingletonMeta):
    def __init__(self):
        self._registry: set[NameSpace] = set()

    def register_namespace(self, namespace: NameSpace) -> None:
        """Safely adds an item to the specific namespace."""
        if namespace in self._registry:
            raise ValueError(f"Existing namespace {namespace}")

        # Note: This does mean that gate (or operation) is never garbage
        # collected until the Namespace exists. This is fine for standard gates.
        self._registry.add(namespace)

        # Default behaviour for user defining his own gates is that namespace is None,
        # Thus those gates are considered temporary by default, we use the same logic in
        # QPE for Controlled Unitary gates, VQA (until Ops i.e. composite gates are introduced).


_GlobalRegistry = _GlobalNameSpaceRegistry()


@dataclass
class NameSpace:
    local_name: str
    parent: NameSpace | None = None
    _registry: set[str] = field(default_factory=set)

    def __post_init__(self):
        if "." in self.local_name:
            raise ValueError(
                f"Namespace local_name '{self.local_name}' cannot contain dots. "
                f"Dots are reserved for hierarchical resolution."
            )
        _GlobalRegistry.register_namespace(self)

    @cached_property
    def name(self) -> str:
        if self.parent:
            return f"{self.parent.name}.{self.local_name}"
        return self.local_name

    def register(self, operation_cls: any) -> None:
        """Safely adds an item to the specific namespace."""
        if operation_cls in self._registry:
            raise NameError(
                f"'{operation_cls.name}' already exists in namespace '{self.name}'"
            )
        self._registry.add(operation_cls)

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return str(self)


NS_STD = NameSpace("std")  # DEFAULT NAMESPACE
NS_GATE = NameSpace("gates", parent=NS_STD)  # Default Gate Namespace
