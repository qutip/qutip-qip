# annotations can be removed when base version is Python 3.14 (PEP 749)
from __future__ import annotations
from functools import cached_property
from dataclasses import dataclass, field


class _SingletonMeta(type):
    """
    Metaclass to implement the Singleton design pattern.
    Note that this is not a thread-safe implementation of a Singleton.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class _GlobalNameSpaceRegistry(metaclass=_SingletonMeta):
    """
    Global registry to manage and store all active namespaces.

    This class enforces a singleton pattern (using the metaclass) to ensure that only one global
    registry exists during the runtime of the application.
    """

    def __init__(self):
        self._registry: set[NameSpace] = set()

    def register_namespace(self, namespace: NameSpace) -> None:
        """
        Safely adds a new namespace to the global registry.

        Note: This means that a gate (or operation) is never garbage
        collected until the Namespace is destroyed. This is the desired
        behavior for standard library gates.

        Parameters
        ----------
        namespace : NameSpace
            The namespace instance to be registered.

        Raises
        ------
        ValueError
            If the namespace already exists within the registry.
        """
        if namespace in self._registry:
            raise ValueError(f"Existing namespace {namespace}")

        self._registry.add(namespace)

        # Default behaviour for user defining his own gates is that namespace is None,
        # Thus those gates are considered temporary by default, we use the same logic in
        # QPE for Controlled Unitary gates, VQA until Ops are implemented.


_GlobalRegistry = _GlobalNameSpaceRegistry()


@dataclass
class NameSpace:
    """
    Represents a distinct, optionally hierarchical namespace for registering
    quantum operations.

    Parameters
    ----------
    local_name : str
        The local identifier for the namespace. Must not contain periods ('.').
    parent : NameSpace or None, optional
        The parent namespace, if this is a nested sub-namespace. Default is None.
    """

    local_name: str
    parent: NameSpace | None = None
    _registry: dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validates the namespace name and registers it globally upon creation.

        Raises
        ------
        ValueError
            If `local_name` contains a dot, as dots are reserved for hierarchy.
        """
        if "." in self.local_name:
            raise ValueError(
                f"Namespace local_name '{self.local_name}' cannot contain dots. "
                f"Dots are reserved for hierarchical resolution."
            )
        _GlobalRegistry.register_namespace(self)

    @cached_property
    def name(self) -> str:
        """
        str: The fully qualified, hierarchical name of the namespace.
        (e.g., 'std.gates').
        """
        if self.parent:
            return f"{self.parent.name}.{self.local_name}"
        return self.local_name

    def register(self, name: str | tuple[str, int, int], operation_cls: any) -> None:
        """
        Safely adds an item to this specific namespace.

        Parameters
        ----------
        name : str or tuple of (str, int, int)
            The identifier for the operation. Use a string for a non-controlled
            gate. Use a tuple `(target_gate.name, num_ctrl_qubits, ctrl_values)`
            as a second key for controlled gates.
        operation_cls : any
            The operation class or object to register.

        Raises
        ------
        NameError
            If an operation with the given name already exists in this namespace.
        """
        existing_gate = self.get(name)
        if existing_gate is not None and existing_gate is not operation_cls:
            raise NameError(
                f"'{operation_cls.name}' already exists in namespace '{self.name}'"
            )
        self._registry[name] = operation_cls

    def get(self, name: str | tuple[str, int, int]) -> any:
        """
        Retrieves a registered item from the namespace.

        Parameters
        ----------
        name : str or tuple of (str, int, int)
            The identifier of the registered operation.

        Returns
        -------
        any
            The registered operation class or object, or None if it is not found.
        """
        if name not in self._registry:
            return None
        return self._registry[name]

    def _remove(self, name: str | tuple[str, int, int]) -> None:
        """
        Removes an item from the namespace registry.

        Parameters
        ----------
        name : str or tuple of (str, int, int)
            The identifier of the operation to remove.

        Raises
        ------
        KeyError
            If the specified name does not exist in the namespace.
        """
        if name not in self._registry:
            raise KeyError(f"{name} does not exists in namespace '{self.name} ")
        del self._registry[name]

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other) -> bool:
        """Checks equality based on the full namespace name."""
        if type(other) is not NameSpace:
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        """Hashes the namespace based on the full namespace name."""
        return hash(self.name)


# NS stands for namespace, std for Standard
NS_STD = NameSpace("std")  # DEFAULT NAMESPACE
NS_GATE = NameSpace("gates", parent=NS_STD)  # Default Gate Namespace
