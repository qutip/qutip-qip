# annotations import won't be needed after minimum version becomes 3.14 (PEP 749)
from __future__ import annotations
import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Type

import numpy as np
from qutip import Qobj
from qutip_qip.operations.namespace import NameSpace

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
        if inspect.isabstract(cls):
            cls._is_frozen = True
            return

        # _is_frozen class attribute (flag) signals class (or subclass) is built,
        # don't overwrite any defaults like num_qubits etc in __setattr__.
        cls._is_frozen = True

        # Namespace being None corresponds to Temporary Gates
        # Only if Namespace is not None register the gate
        if (namespace := getattr(cls, "namespace", None)) is not None:

            # We are checking beforehand because in case of Controlled Gate
            # two key's refer to the same controlled gate:
            # gate_name, (target_gate.name, num_ctrl_qubits, ctrl_value)

            # If suppose (target_gate=X, num_ctrl_qubits=1, ctrl_value=0) existed
            # but we were redefining it with a different name, the cls.name insert
            # step would go through, but wrt. second key won't and will throw an error.
            # This will lead to leakage in the namespace i.e. classes which don't exist but are in the namespace.

            existing_gate = namespace.get(cls.name)
            if existing_gate is not None:
                try:
                    # Check if both classes originate from the exact same physical file.
                    # If they do, this is a namespace alias/reload
                    # This is needed because qutip.qip import support is still needed
                    if inspect.getfile(existing_gate) == inspect.getfile(cls):
                        return
                    else:
                        raise ValueError(
                            f"Existing {cls.name} in namespace {namespace}"
                        )
                except TypeError:
                    pass  # Fallback

            # The basic principle is don't define a gate class if it already exists
            if cls.is_controlled():
                cls.namespace.register(
                    (
                        cls.target_gate.name,
                        cls.num_ctrl_qubits,
                        cls.ctrl_value,
                    ),
                    cls,
                )
            cls.namespace.register(cls.name, cls)

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

        if cls.__dict__.get("_is_frozen", False) and name in _read_only_set:
            raise AttributeError(f"{name} is read-only!")
        super().__setattr__(name, value)

    def __str__(cls) -> str:
        # Don't remove these getattr statement otherwise doctest will fail
        # The reason is because how Sphinx builds and runs doctest even on abstract classes.
        gatename = getattr(cls, "name", cls.__name__)
        return f"Gate({gatename})"

    def __repr__(cls) -> str:
        gatename = getattr(cls, "name", cls.__name__)
        num_qubits = getattr(cls, "num_qubits", None)
        return f"Gate({gatename}, num_qubits={num_qubits})"


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
    namespace: NameSpace | None = None

    name: str
    num_qubits: int
    self_inverse: bool = False
    is_clifford: bool = False
    latex_str: str

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Automatically runs when a new subclass is defined via inheritance.

        This method sets the ``name`` and ``latex_str`` attributes if
        they are not defined in the subclass. It also validates that ``num_qubits``
        is a non-negative integer, ``is_clifford``, ``self_inverse`` are
        bool and ``inverse`` method is not defined if ``self_inverse`` is set True.
        """

        # Skip the below check for an abstract class
        if inspect.isabstract(cls):
            return super().__init_subclass__(**kwargs)

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
                f"Class '{cls.name}' attribute 'num_qubits' must be a non-negative integer, "
                f"got {type(num_qubits)} with value {num_qubits}."
            )

        # Check is_clifford is a bool
        if type(cls.is_clifford) is not bool:
            raise TypeError(
                f"Class '{cls.name}' attribute 'is_clifford' must be a bool, "
                f"got {type(cls.is_clifford)} with value {cls.is_clifford}."
            )

        # Check self_inverse is a bool
        if type(cls.self_inverse) is not bool:
            raise TypeError(
                f"Class '{cls.name}' attribute 'self_inverse' must be a bool, "
                f"got {type(cls.self_inverse)} with value {cls.self_inverse}."
            )

        # Can't define inverse() method if self_inverse is set True
        if cls.self_inverse and "inverse" in cls.__dict__:
            raise TypeError(
                f"Gate '{cls.name}' is marked as self_inverse=True. "
                f"You are not allowed to override the 'inverse()' method. "
                f"Remove the method; the base class handles it automatically."
            )

        try:
            param_flag = cls.is_parametric()
        except TypeError as e:
            raise TypeError(
                f"Class '{cls.name}' must define 'is_parametric()' as a callable "
                f"@staticmethod or @classmethod taking no instance arguments. "
                f"Error: {e}"
            )

        if type(param_flag) is not bool:
            raise TypeError(
                f"Class '{cls.name}' method 'is_controlled()' must return a strict bool, "
                f"got {type(param_flag)} with value {param_flag}."
            )

        try:
            control_flag = cls.is_controlled()
        except TypeError as e:
            raise TypeError(
                f"Class '{cls.name}' must define 'is_parametric()' as a callable "
                f"@staticmethod or @classmethod taking no instance arguments. "
                f"Error: {e}"
            )

        if type(control_flag) is not bool:
            raise TypeError(
                f"Class '{cls.name}' method 'is_controlled()' must return a strict bool, "
                f"got {type(control_flag)} with value {control_flag}."
            )

        return super().__init_subclass__(**kwargs)

    def __init__(self) -> None:
        """
        This method is overwritten in case of Parametrized and Controlled Gates.
        """
        raise TypeError(
            f"Gate '{self.name}' can't be initialised. "
            f"If your gate requires parameters, it must inherit from 'ParametricGate'. "
            f"Or if it must be controlled, it must inherit from 'ControlledGate'."
        )

    @staticmethod
    @abstractmethod
    def get_qobj(dtype: str = "dense") -> Qobj:
        """
        Get the :class:`qutip.Qobj` representation of the gate operator.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        raise NotImplementedError

    @classmethod
    def inverse(cls) -> Type[Gate]:
        """
        Return the inverse of the gate.

        If ``self_inverse`` is True, returns ``self``. Otherwise,
        returns the specific inverse gate class.

        Returns
        -------
        Type[Gate]
            A Gate instance representing $G^{-1}$.
        """
        if cls.self_inverse:
            return cls
        return get_unitary_gate(f"{cls.name}_inv", cls.get_qobj().dag())

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
    gate_name: str, U: Qobj, gate_namespace: NameSpace | None = None
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
        def get_qobj(dtype=U.dtype):
            return U.to(dtype)

    return _CustomGate
