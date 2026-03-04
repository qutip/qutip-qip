# annotations import won't be needed after minimum version becomes 3.14 (PEP 749)
from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
import inspect
from typing import Type

import numpy as np
from qutip import Qobj
from qutip_qip.operations.namespace import NameSpace, NS_USER

_read_only_set: set[str] = set(
    (
        "namespace",
        "num_qubits",
        "num_ctrl_qubits",
        "num_params",
        "ctrl_value",
        "self_inverse",
        "is_clifford",
        "target_gate",
        "latex_str",
    )
)


class _GateMetaClass(ABCMeta):
    _registry: dict[str, set] = {}

    def __init__(cls, name, bases, attrs):
        """
        This method is automatically invoked during class creation. It validates that
        the new gate class has a unique name within its specific namespace (defaulting
        to "std"). If the same gate already exists in that namespace, it raises a strict
        TypeError to prevent ambiguous gate definitions.

        This is required since in the codebase at several places like decomposition we
        check for e.g. gate.name == 'X', which is corrupted if user defines a gate with
        the same name.
        """
        super().__init__(name, bases, attrs)

        # Don't register the Abstract Gate Classes or private helpers
        if inspect.isabstract(cls) or name.startswith("_"):
            cls._is_frozen = True
            return

        # _is_frozen class attribute (flag) signals class (or subclass) is built,
        # don't overwrite any defaults like num_qubits etc in __setattr__.
        cls._is_frozen = True

        # We obviously don't register abstract classes to the namespace.
        namespace = cls.namespace
        namespace.register(cls.name, cls)

    def __setattr__(cls, name: str, value: any) -> None:
        """
        One of the main purpose of this meta class is to enforce read-only constraints
        on specific class attributes. This prevents critical attributes from being
        overwritten after definition, while still allowing them to be set during inheritance.

        For example:
            class X(Gate):
                num_qubits = 1   # Allowed (during class creation)

        But:
            X.num_qubits = 2     # Raises AttributeError (prevention of overwrite)

        This is required since num_qubits etc. are class attributes (shared by all object instances).
        """
        # cls.__dict__.get() instead of getattr() ensures we don't
        # accidentally inherit the True flag from a parent class for _is_frozen.
        if (
            cls.__dict__.get("_is_frozen", False)
            and name in cls._read_only_set
        ):
            raise AttributeError(f"{name} is read-only!")
        super().__setattr__(name, value)

    def __str__(cls) -> str:
        return f"Gate({cls.name})"

    def __repr__(cls) -> str:
        return f"Gate({cls.name}, num_qubits={cls.num_qubits})"

    def __eq__(cls, other: any) -> bool:
        # Return False if other is not a class.
        if not isinstance(other, type):
            return False

        if cls is other:
            return True

        if isinstance(other, _GateMetaClass):
            cls_name = getattr(cls, "name", None)
            other_name = getattr(other, "name", None)

            cls_namespace = getattr(cls, "namespace", None)
            other_namespace = getattr(other, "namespace", None)

            # They are equal if they share the same name and namespace
            return cls_name == other_name and cls_namespace == other_namespace

        return False

    def __hash__(cls) -> int:
        """
        Required because __eq__ is overridden.
        Hashes the class based on its unique identity (namespace and name)
        so it can still be safely used in the _registry sets and dicts.
        """
        return hash(
            (getattr(cls, "namespace", "std"), getattr(cls, "name", None))
        )

    def clear_cache(cls, namespace: str):
        """
        Clears the gate class registry based on the namespace.

        Parameters
        ----------
        namespace : str, optional
            If provided, only clears gates belonging to this namespace
            (e.g., 'custom'). If None, clears ALL gates (useful for hard resets).
        """
        if namespace == "std":
            raise ValueError("Can't clear std Gates")
        else:
            cls._registry[namespace] = set()


class Gate(ABC, metaclass=_GateMetaClass):
    r"""
    Abstract base class for a quantum gate.

    Concrete gate classes or gate implementations should be defined as subclasses
    of this class.

    Attributes
    ----------
    name : str
        The name of the gate. If not manually set, this defaults to the
        class name. This is a class attribute; modifying it affects all
        instances.

    num_qubits : int
        The number of qubits the gate acts upon. This is a mandatory
        class attribute for subclasses.

    self_inverse: bool
        Indicates if the gate is its own inverse (e.g., $U = U^{-1}$).
        Default value is False.

    is_clifford: bool
        Indicates if the gate belongs to the Clifford group, which maps
        Pauli operators to Pauli operators. Default value is False

    latex_str : str
        The LaTeX string representation of the gate (used for circuit drawing).
        Defaults to the class name if not provided.
    """

    # __slots__ in Python are meant to fixed-size array of attribute values
    # instead of a default dynamic sized __dict__ created in object instances.
    # This helps save memory, faster lookup time & restrict adding new attributes to class.
    __slots__ = ()
    namespace: NameSpace

    name: str
    num_qubits: int
    self_inverse: bool = False
    is_clifford: bool = False
    latex_str: str

    def __init_subclass__(cls, **kwargs):
        """
        Automatically runs when a new subclass is defined via inheritance.

        This method sets the ``name`` and ``latex_str`` attributes
        if they are not defined in the subclass. It also validates that
        ``num_qubits`` is a non-negative integer.
        """

        super().__init_subclass__(**kwargs)

        # Skip the below check for an abstract class
        if inspect.isabstract(cls):
            return

        if getattr(cls, "namespace", None) is None:
            raise AttributeError(
                f"Missing attribute namespace from {cls.__name__}"
            )

        # If name attribute in subclass is not defined, set it to the name of the subclass
        # e.g. class H(Gate):
        #         pass

        #      print(H.name) -> 'H'

        # e.g. class H(Gate):
        #         name = "Hadamard"
        #         pass

        #      print(H.name) -> 'Hadamard'

        if "name" not in vars(cls):
            cls.name = cls.__name__

        # Same as above for attribute latex_str (used in circuit draw)
        if "latex_str" not in vars(cls):
            cls.latex_str = cls.__name__

        # Assert num_qubits is a non-negative integer
        num_qubits = getattr(cls, "num_qubits", None)
        if (type(num_qubits) is not int) or (num_qubits < 0):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_qubits' must be a non-negative integer, "
                f"got {type(num_qubits)} with value {num_qubits}."
            )

    def __init__(self) -> None:
        """
        This method is overwritten by Parametrized and Controlled Gates.
        """
        raise TypeError(
            f"Gate '{type(self).name}' can't be initialized. "
            f"If your gate requires parameters, it must inherit from 'ParametricGate'. "
            f"Or if it must be controlled and needs control_value, it must inherit from 'ControlledGate'."
        )

    @staticmethod
    @abstractmethod
    def get_qobj() -> Qobj:
        """
        Get the :class:`qutip.Qobj` representation of the gate operator.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        pass

    @classmethod
    def inverse(cls) -> Type[Gate]:
        """
        Return the inverse of the gate.

        If ``self_inverse`` is True, returns ``self``. Otherwise,
        returns the specific inverse gate class.

        Returns
        -------
        Gate
            A Gate instance representing $G^{-1}$.
        """
        if cls.self_inverse:
            return cls
        raise NotImplementedError

    @staticmethod
    def is_controlled() -> bool:
        """
        Check if the gate is a controlled gate.

        Returns
        -------
        bool
        """
        return False

    @staticmethod
    def is_parametric() -> bool:
        """
        Check if the gate accepts variable parameters (e.g., rotation angles).

        Returns
        -------
        bool
            True if the gate is parametric (e.g., RX, RY, RZ), False otherwise.
        """
        return False


def get_unitary_gate(
    gate_name: str, U: Qobj, gate_namespace: NameSpace = NS_USER
) -> Type[Gate]:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    # Check whether U is unitary
    n = np.log2(U.shape[0])
    if n != np.log2(U.shape[1]):
        raise ValueError("The U must be square matrix.")

    if n % 1 != 0:
        raise ValueError("The unitary U must have dim NxN, where N=2^n")

    if not np.allclose((U * U.dag()).full(), np.eye(U.shape[0])):
        raise ValueError("U must be a unitary matrix")

    class _CustomGate(Gate):
        __slots__ = ()
        namespace = gate_namespace
        name = gate_name
        num_qubits = int(n)
        self_inverse = U == U.dag()

        @staticmethod
        def get_qobj():
            return U

    return _CustomGate
