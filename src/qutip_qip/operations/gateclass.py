# annotations import won't be needed after minimum version becomes 3.14 (PEP 749)
from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
import inspect

import numpy as np
from qutip import Qobj


class _GateMetaClass(ABCMeta):
    _registry: dict[str, set] = {}
    _read_only_set: set[str] = set(
        (
            "num_qubits",
            "num_ctrl_qubits",
            "num_params",
            "self_inverse",
            "is_clifford",
            "target_gate",
            "latex_str",
        )
    )

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
            return

        namespace = attrs.get("namespace", "std")
        name = cls.name

        if namespace not in cls._registry:
            cls._registry[namespace] = set()

        if name in cls._registry[namespace]:
            raise TypeError(
                f"Gate Conflict: '{name}' is already defined in namespace '{namespace}' "
            )
        cls._registry[namespace].add(name)

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

    def __call__(cls, *args, **kwargs):
        """
        So creating CNOT(control_value=1) 10,000 times (e.g., for a large circuit) becomes instant
        because it's just a dictionary lookup after the first time. Also in memory
        we only need to store one copy of CNOT gate, no matter how many times it appears
        in the circuit.
        """
        return super().__call__(*args, **kwargs)

        # For RX(0.5), RX(0.1) we want different instances.
        # Same for CX(control_value=0), CX(control_value=1)
        # TODO This needs to be implemented efficiently

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
        # Check if the attribute is in our protected set
        # Using cls.__dict__ ignores parent classes, allowing __init_subclass__ 
        # to set it once for the child class.
        if name in cls._read_only_set and name in cls.__dict__:
            raise AttributeError(f"{name} is read-only!")
        super().__setattr__(name, value)


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

    _namespace: str = "std"
    num_qubits: int
    self_inverse: bool = False
    is_clifford: bool = False

    name = None
    latex_str = None

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
    def inverse(cls) -> Gate:
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
    def is_parametric_gate() -> bool:
        """
        Check if the gate accepts variable parameters (e.g., rotation angles).

        Returns
        -------
        bool
            True if the gate is parametric (e.g., RX, RY, RZ), False otherwise.
        """
        return False

    def __str__(self) -> str:
        return f"Gate({self.name})"

    def __repr__(self) -> str:
        return f"Gate({self.name}, num_qubits={self.num_qubits}, qobj={self.get_qobj()})"


def unitary_gate(gate_name: str, U: Qobj, namespace: str = "custom") -> Gate:
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
        _namespace = namespace
        name = gate_name
        num_qubits = int(n)
        self_inverse = (U == U.dag())

        @staticmethod
        def get_qobj():
            return U

    return _CustomGate
