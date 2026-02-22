import inspect
from abc import abstractmethod

from qutip import Qobj
from qutip_qip.operations import Gate, ParametricGate, controlled_gate_unitary


class ControlledGate(Gate):
    r"""
    Abstract base class for controlled quantum gates.

    A controlled gate applies a target unitary operation only when the control
    qubits are in a specific state.

    Parameters
    ----------
    control_value : int, optional
        The decimal value of the control state required to execute the
        unitary operator on the target qubits.

        Examples:
        * If the gate should execute when the 0-th qubit is $|1\rangle$,
            set ``control_value=1``.
        * If the gate should execute when two control qubits are $|10\rangle$
            (binary 10), set ``control_value=0b10``.

        Defaults to all-ones (e.g., $2^N - 1$) if not provided.

    Attributes
    ----------
    num_ctrl_qubits : int
        The number of qubits acting as controls.

    target_gate : Gate
        The gate to be applied to the target qubits.
    """

    __slots__ = ("arg_value", "arg_label", "_control_value")
    num_ctrl_qubits: int
    target_gate: Gate

    def __init_subclass__(cls, **kwargs):
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

        # Automatically copy the validator from the target
        if hasattr(cls.target_gate, "validate_params"):
            cls.validate_params = staticmethod(cls.target_gate.validate_params)

        # Default set_inverse
        cls.self_inverse = cls.target_gate.self_inverse

        # In the circuit plot, only the target gate is shown.
        # The control has its own symbol.
        cls.latex_str = cls.target_gate.latex_str

    def __init__(
        self,
        arg_value: any = None,
        control_value: int | None = None,
        arg_label: str | None = None,
    ) -> None:
        if control_value is not None:
            self._validate_control_value(control_value)
            self._control_value = control_value
        else:
            self._control_value = (2**self.num_ctrl_qubits) - 1

        if self.is_parametric_gate():
            ParametricGate.__init__(
                self, arg_value=arg_value, arg_label=arg_label
            )

    @property
    @abstractmethod
    def target_gate() -> Gate:
        pass

    @property
    def self_inverse(self) -> int:
        return self.target_gate.self_inverse

    @property
    def control_value(self) -> int:
        return self._control_value

    @classmethod
    def _validate_control_value(cls, control_value: int) -> None:
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

        if type(control_value) is not int:
            raise TypeError(
                f"Control value must be an int, got {control_value}"
            )

        if control_value < 0 or control_value > 2**cls.num_ctrl_qubits - 1:
            raise ValueError(
                f"Control value can't be negative and can't be greater than "
                f"2^num_ctrl_qubits - 1, got {control_value}"
            )

    def get_qobj(self) -> Qobj:
        """
        Construct the full Qobj representation of the controlled gate.

        Returns
        -------
        qobj : qutip.Qobj
            The unitary matrix representing the controlled operation.
        """
        target_gate = self.target_gate
        if self.is_parametric_gate():
            target_gate = target_gate(self.arg_value)

        return controlled_gate_unitary(
            U=target_gate.get_qobj(),
            num_controls=self.num_ctrl_qubits,
            control_value=self.control_value,
        )

    @staticmethod
    def is_controlled_gate() -> bool:
        return True

    @classmethod
    def is_parametric_gate(cls) -> bool:
        return cls.target_gate.is_parametric_gate()

    def __str__(self) -> str:
        return f"Gate({self.name}, target_gate={self.target_gate}, num_ctrl_qubits={self.num_ctrl_qubits}, control_value={self.control_value})"

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False

        if self.control_value != other.control_value:
            return False
        return True


def controlled_gate(
    gate: Gate,
    n_ctrl_qubits: int = 1,
    gate_name: str = "_CustomControlledGate",
    namespace: str = "custom",
) -> ControlledGate:
    """
    Gate Factory for Controlled Gate that takes a gate and num_ctrl_qubits.
    """

    class _CustomControlledGate(ControlledGate):
        __slots__ = ()
        _namespace = namespace
        name = gate_name
        num_qubits = n_ctrl_qubits + gate.num_qubits
        num_ctrl_qubits = n_ctrl_qubits
        target_gate = gate
        latex_str = rf"C{gate.name}"

    return _CustomControlledGate
