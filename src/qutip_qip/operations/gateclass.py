# annotations import won't be needed after minimum version becomes 3.14 (PEP 749)
from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from typing import ClassVar

from qutip import Qobj
from qutip_qip.operations.helper import (
    cached_class_or_instance_method,
    class_or_instance_method,
)
from qutip_qip.operations.op import Op, is_op, op_factory
from qutip_qip.typing import Real


class Gate(Op):
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

    num_qreg : int
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

    config = Op.config | {"is_clifford", "latex_str"}
    is_clifford: ClassVar[bool] = False
    latex_str: ClassVar[str]

    def __init_subclass__(cls) -> None:
        """
        Automatically runs when a new subclass is defined via inheritance.

        This method sets the name and latex_str attributes if
        they are not defined in the subclass. It also validates that is_clifford, self_inverse are
        bool and inverse method is not defined if self_inverse is set True.
        """

        # Skip the below check for an abstract class
        super().__init_subclass__()

        if "latex_str" not in vars(cls):
            cls.latex_str = cls.name

        if "dtype" not in inspect.signature(cls.compute_qobj).parameters:
            raise TypeError(f"{cls.name}.compute_qobj must accept a dtype parameter.")

        if not isinstance(cls.is_clifford, bool):
            raise TypeError(f"{cls.name}: is_clifford must be a bool.")

        # Can't define inverse() method if self_inverse is set True
        if cls.self_inverse and "inverse" in vars(cls):
            raise TypeError(
                f"Gate '{cls.name}' is marked as self_inverse=True. "
                f"You are not allowed to override the 'inverse()' method. "
                f"Remove the method; the base class handles it automatically."
            )

        # Ideally we don't require cls.is_derived here (num_qreg, num_creg etc.) are parameteric
        # But we could have a case in future where say MCX(n) or QFT(n) is a basis gate on some future modality
        # This allows us to handle those cases as well.
        if inspect.isabstract(cls) or cls.is_derived:
            return

        cls.num_qubits = cls.num_qreg  # TODO: delete this

    @abstractmethod
    def compute_qobj(self, dtype: str = "dense") -> Qobj:
        """Build the gate's unitary. User's need to implement this"""
        raise NotImplementedError

    @cached_class_or_instance_method
    def get_qobj(cls_ins, dtype: str = "dense") -> Qobj:
        """
        Get the :class:`qutip.Qobj` representation of the gate operator.

        Returns
        -------
        qobj : :obj:`qutip.Qobj`
            The compact gate operator as a unitary matrix.
        """
        if not is_op(cls_ins):
            raise TypeError(
                f"{cls_ins.name} is parametric, build an instance first "
                f"e.g. {cls_ins.name}(...).get_qobj()."
            )
        return cls_ins.compute_qobj(dtype)

    @class_or_instance_method
    def inverse(cls_ins) -> Gate | type[Gate]:
        """
        Return the inverse of the gate.

        If ``self_inverse`` is True, returns ``self``. Otherwise,
        returns the specific inverse gate class.

        Returns
        -------
        type[Gate]
            A Gate instance representing $G^{-1}$.
        """
        if cls_ins.self_inverse:
            return cls_ins
        return get_unitary_gate(f"{cls_ins.name}_inv", cls_ins.get_qobj().dag())

    @property
    def arg_value(self) -> tuple:  # deprecated alias for the 72 call sites
        warnings.warn(
            "arg_value argument for parametric gates is deprecated",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.params


class AngleParametricGate(Gate):
    """Parametric gate whose parameters are all real angles."""

    def validate_params(self) -> None:
        for v in self.params:
            if isinstance(v, bool) or not isinstance(
                v, Real
            ):  # bool is an int in Python, but not an angle from our perspective
                raise TypeError(f"{self.name}: parameters must be real, got {v!r}.")


@op_factory
def get_unitary_gate(name: str, U: Qobj) -> type[Gate]:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    _name = name
    if not U.isoper or U.dims[0] != U.dims[1]:
        raise ValueError(
            "U must be a square operator with equal input and output dims."
        )

    if not U.isunitary:
        raise ValueError("U must be unitary.")

    class _CustomGate(Gate):
        name = _name
        num_qreg = len(U.dims[0])
        qreg_dim = tuple(U.dims[0])
        self_inverse = U == U.dag()

        @staticmethod
        def compute_qobj(dtype=U.dtype):
            return U.to(dtype)

    _CustomGate.__name__ = _CustomGate.__qualname__ = _name
    return _CustomGate
