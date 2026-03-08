import inspect
import warnings
from typing import Type
from functools import partial
from abc import abstractmethod

from qutip import Qobj
from qutip_qip.operations import (
    Gate,
    NameSpace,
    controlled_gate_unitary,
)


class class_or_instance_method:
    """
    Binds a method to the instance if called on an instance,
    or to the class if called on the class.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        # Called on the class (e.g., CX.get_qobj())
        if instance is None:
            return partial(self.func, owner)

        # Called on the instance (e.g., CRX(0.5).get_qobj())
        return partial(self.func, instance)


class ControlledGate(Gate):
    r"""
    Abstract base class for controlled quantum gates.

    A controlled gate applies a target unitary operation only when the control
    qubits are in a specific state.

    Attributes
    ----------
    target_gate : :class:`.Gate`
        The gate to be applied to the target qubits.

    num_ctrl_qubits : int
        The number of qubits acting as controls.

    ctrl_value : int
        The decimal value of the control state required to execute the
        unitary operator on the target qubits.

        Examples:
        * If the gate should execute when the 0-th qubit is $|1\rangle$,
            set ``ctrl_value=1``.
        * If the gate should execute when two control qubits are $|10\rangle$
            (binary 10), set ``ctrl_value=0b10``.
    """

    __slots__ = ("_target_inst",)

    num_ctrl_qubits: int
    ctrl_value: int
    target_gate: Type[Gate]

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Validates the subclass definition.
        """

        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        # Must have a target_gate
        target_gate = getattr(cls, "target_gate", None)
        if target_gate is None or not issubclass(target_gate, Gate):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'target_gate' must be a Gate class, "
                f"got {type(target_gate)} with value {target_gate}."
            )

        # Check num_ctrl_qubits is a positive integer
        num_ctrl_qubits = getattr(cls, "num_ctrl_qubits", None)
        if (type(num_ctrl_qubits) is not int) or (num_ctrl_qubits < 1):
            raise TypeError(
                f"Class '{cls.__name__}' attribute 'num_ctrl_qubits' must be a postive integer, "
                f"got {type(num_ctrl_qubits)} with value {num_ctrl_qubits}."
            )

        # Check num_ctrl_qubits < num_qubits
        if not cls.num_ctrl_qubits < cls.num_qubits:
            raise ValueError(
                f"{cls.__name__}: 'num_ctrl_qubits' must be less than the 'num_qubits'"
            )

        # Check num_ctrl_qubits + target_gate.num_qubits = num_qubits
        if cls.num_ctrl_qubits + cls.target_gate.num_qubits != cls.num_qubits:
            raise AttributeError(
                f"'num_ctrls_qubits' {cls.num_ctrl_qubits} + 'target_gate qubits' {cls.target_gate.num_qubits} must be equal to 'num_qubits' {cls.num_qubits}"
            )
        cls._validate_control_value()

        # Default self_inverse
        # Don't replace cls.__dict__ with hasattr() that does a MRO search
        if "self_inverse" not in cls.__dict__:
            cls.self_inverse = cls.target_gate.self_inverse

        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        if "latex_str" not in cls.__dict__:
            cls.latex_str = cls.target_gate.latex_str

    def __init__(self, *args, **kwargs) -> None:
        self._target_inst = self.target_gate(*args, **kwargs)

    def __getattr__(self, name: str) -> any:
        """
        If an attribute (like 'arg_value') or method (like 'validate_params')
        isn't found on the ControlledGate, Python falls back to this method.
        We forward the request to the underlying target gate instance.
        """
        return getattr(self._target_inst, name)

    def __setattr__(self, name, value) -> None:
        """
        Intercept attribute assignment. If it's our internal storage variable,
        set it normally on this instance. Otherwise, forward the assignment
        to the underlying target gate.
        """
        if name == "_target_inst":
            super().__setattr__(name, value)
        else:
            setattr(self._target_inst, name, value)

    # Although target_gate is specified as a class attribute, It has been
    # been made an abstract method to make ControlledGate abstract (required in Metaclass)
    # This is because Python currently doesn't support abstract class attributes.
    @property
    @abstractmethod
    def target_gate() -> Type[Gate]:
        pass

    @classmethod
    def _validate_control_value(cls) -> None:
        """
        Internal validation for the control value.

        Raises
        ------
        TypeError
            If ctrl_value is not an integer.
        ValueError
            If ctrl_value is negative or exceeds the maximum value
            possible for the number of control qubits ($2^N - 1$).
        """

        if type(cls.ctrl_value) is not int:
            raise TypeError(
                f"Control value must be an int, got {cls.ctrl_value}"
            )

        if cls.ctrl_value < 0 or cls.ctrl_value > 2**cls.num_ctrl_qubits - 1:
            raise ValueError(
                f"Control value can't be negative and can't be greater than "
                f"2^num_ctrl_qubits - 1, got {cls.ctrl_value}"
            )

    @class_or_instance_method
    def get_qobj(cls_or_self) -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        if isinstance(cls_or_self, type):
            return controlled_gate_unitary(
                U=cls_or_self.target_gate.get_qobj(),
                num_controls=cls_or_self.num_ctrl_qubits,
                control_value=cls_or_self.ctrl_value,
            )

        return controlled_gate_unitary(
            U=cls_or_self._target_inst.get_qobj(),
            num_controls=cls_or_self.num_ctrl_qubits,
            control_value=cls_or_self.ctrl_value,
        )

    @class_or_instance_method
    def inverse(cls_or_self) -> Gate | Type[Gate]:
        if cls_or_self.self_inverse:
            inverse_gate = cls_or_self

        # Non-parametrized Gates e.g. S
        elif isinstance(cls_or_self, type):
            inverse_gate = controlled(
                cls_or_self.target_gate.inverse(),
                cls_or_self.num_ctrl_qubits,
                cls_or_self.ctrl_value,
            )

        else:
            inverse_gate_class, param = cls_or_self._target_inst.inverse(
                expanded=True
            )

            inverse_gate = controlled(
                inverse_gate_class,
                cls_or_self.num_ctrl_qubits,
                cls_or_self.ctrl_value,
            )(*param)

        return inverse_gate

    @staticmethod
    def is_controlled() -> bool:
        return True

    @classmethod
    def is_parametric(cls) -> bool:
        return cls.target_gate.is_parametric()

    @classmethod
    def __str__(cls) -> str:
        return f"Gate({cls.name}, target_gate={cls.target_gate}, num_ctrl_qubits={cls.num_ctrl_qubits}, control_value={cls.ctrl_value})"

    def __eq__(self, other) -> bool:
        # Returns false for CRX(0.5), CRY(0.5)
        if type(self) is not type(other):
            return False

        # Returns false for CRX(0.5), CRX(0.6)
        if self.is_parametric() and self._target_inst != other._target_inst:
            return False

        return True

    def __hash__(self) -> int:
        return super().__hash__()


def controlled(
    gate: Type[Gate],
    n_ctrl_qubits: int = 1,
    control_value: int | None = None,
    gate_name: str | None = None,
    gate_namespace: NameSpace | None = None,
) -> ControlledGate:
    """
    Gate Factory for Controlled Gate that takes a gate and num_ctrl_qubits.
    """

    if gate_namespace is None:
        gate_namespace = gate.namespace

    if control_value is None:
        control_value = 2**n_ctrl_qubits - 1

    if gate_name is None:
        gate_name = f"{'C' * n_ctrl_qubits}{gate.name}"

    if gate_namespace is not None:
        found_gate = gate_namespace.get(
            (gate.name, n_ctrl_qubits, control_value)
        )
        if found_gate is not None:
            warnings.warn(
                f"Found the same existing Controlled Gate {found_gate.name}",
                UserWarning,
            )
            return found_gate

    class _CustomControlledGate(ControlledGate):
        __slots__ = ()
        namespace = gate_namespace
        name = gate_name
        num_qubits = n_ctrl_qubits + gate.num_qubits
        num_ctrl_qubits = n_ctrl_qubits
        ctrl_value = control_value
        target_gate = gate
        latex_str = r"{gate_name}"

    return _CustomControlledGate
