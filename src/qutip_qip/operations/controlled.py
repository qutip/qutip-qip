import inspect
from typing import Type
from functools import partial
from abc import abstractmethod

from qutip import Qobj
from qutip_qip.operations import Gate, controlled_gate_unitary


class class_or_instance_method:
    """
    Binds a method to the instance if called on an instance,
    or to the class if called on the class.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return partial(self.func, owner) # Called on the class (e.g., CX.get_qobj())
        
        return partial(self.func, instance) # Called on the instance (e.g., CRX(0.5).get_qobj())


class ControlledGate(Gate):
    r"""
    Abstract base class for controlled quantum gates.

    A controlled gate applies a target unitary operation only when the control
    qubits are in a specific state.

    Attributes
    ----------
    num_ctrl_qubits : int
        The number of qubits acting as controls.

    target_gate : Gate
        The gate to be applied to the target qubits.

    ctrl_value : int
        The decimal value of the control state required to execute the
        unitary operator on the target qubits.

        Examples:
        * If the gate should execute when the 0-th qubit is $|1\rangle$,
            set ``ctrl_value=1``.
        * If the gate should execute when two control qubits are $|10\rangle$
            (binary 10), set ``ctrl_value=0b10``.
    """

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

    @property
    @abstractmethod
    def target_gate() -> Gate:
        pass

    @property
    def self_inverse(self) -> int:
        return self.target_gate.self_inverse

    @classmethod
    def _validate_control_value(cls) -> None:
        """
        Internal validation for the control value.

        Raises
        ------
        TypeError
            If control_value is not an integer.
        ValueError
            If control_value is negative or exceeds the maximum value
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
    def get_qobj(self_or_cls) -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        if isinstance(self_or_cls, type):
            return controlled_gate_unitary(
                U=self_or_cls.target_gate.get_qobj(),
                num_controls=self_or_cls.num_ctrl_qubits,
                control_value=self_or_cls.ctrl_value,
            )
        
        return controlled_gate_unitary(
            U=self_or_cls._target_inst.get_qobj(),
            num_controls=self_or_cls.num_ctrl_qubits,
            control_value=self_or_cls.ctrl_value,
        )


    def inverse_gate(self) -> Gate:
        if not self.is_parametric():
            return controlled(
                self.target_gate.inverse_gate(),
                self.num_ctrl_qubits,
                self.ctrl_value
            )

        # else:
        #     inverse_target_gate = self.target_gate(self.arg_value).inverse_gate()
        #     arg_value = inverse_target_gate.arg_value
        #     inverse_gate = controlled(
        #         type(inverse_target_gate), self.num_ctrl_qubits, self.ctrl_value
        #     )

        #     return inverse_gate(arg_value=arg_value)

    @staticmethod
    def is_controlled() -> bool:
        return True

    @classmethod
    def is_parametric(cls) -> bool:
        return cls.target_gate.is_parametric()

    @classmethod
    def __str__(cls) -> str:
        return f"Gate({cls.name}, target_gate={cls.target_gate}, num_ctrl_qubits={cls.num_ctrl_qubits}, control_value={cls.ctrl_value})"


def controlled(
    gate: Gate,
    n_ctrl_qubits: int = 1,
    control_value: int | None = None,
    gate_name: str | None = None,
    namespace: str = "custom",
) -> ControlledGate:
    """
    Gate Factory for Controlled Gate that takes a gate and num_ctrl_qubits.
    """

    if gate_name is None:
        gate_name = f"C{gate.name}"

    if control_value is None:
        control_value = 2**n_ctrl_qubits - 1

    class _CustomControlledGate(ControlledGate):
        __slots__ = ()
        _namespace = namespace
        name = gate_name
        num_qubits = n_ctrl_qubits + gate.num_qubits
        num_ctrl_qubits = n_ctrl_qubits
        ctrl_value = control_value
        target_gate = gate
        latex_str = r"{gate_name}"

    return _CustomControlledGate
