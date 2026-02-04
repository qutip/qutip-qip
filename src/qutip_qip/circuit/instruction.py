from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from qutip_qip.operations import Gate, Measurement, ParametrizedGate


def _validate_non_negative_int_tuple(T, txt="qubit"):
    if type(T) is not tuple:
        raise ValueError(f"Must pass a tuple for {txt}, got {type(T)}")

    for q in T:
        if not isinstance(q, int):
            raise ValueError(f"All {txt} indices must be an int, found {q}")

        if q < 0:
            raise ValueError(f"{txt} indices must be non-negative, found {q}")


@dataclass(frozen=True)
class CircuitInstruction(ABC):
    operation: Gate | Measurement
    qubits: tuple[int] = tuple()
    cbits: tuple[int] = tuple()
    style: dict = field(default_factory=dict)

    def __post_init__(self):
        """Basic validation for all instructions."""
        if not len(self.qubits) and not len(self.cbits):
            raise ValueError(
                "Circuit Instruction must operate on at least one qubit or cbit."
            )

        _validate_non_negative_int_tuple(self.qubits, "qubit")
        _validate_non_negative_int_tuple(self.cbits, "cbit")

        if len(self.qubits) != len(set(self.qubits)):
            raise ValueError("Found repeated qubits")

        if len(self.cbits) != len(set(self.cbits)):
            raise ValueError("Found repeated cbits")

    def is_gate_instruction(self) -> bool:
        return False

    def is_measurement_instruction(self) -> bool:
        return False

    @abstractmethod
    def to_qasm(self, qasm_out):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class GateInstruction(CircuitInstruction):
    operation: Gate
    cbits_ctrl_value: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.is_gate_instruction():
            raise ValueError(f"Operation must be a Gate, got {self.operation}")

        if len(self.qubits) != self.operation.num_qubits:
            raise ValueError(
                f"Gate '{self.operation.name}' requires {self.operation.num_qubits} qubits."
                f" But got {len(self.qubits)}."
            )

        if self.cbits_ctrl_value is not None:
            if self.cbits_ctrl_value < 0:
                raise ValueError(
                    f"Classical Control value can't be negative, got {self.cbits_ctrl_value}"
                )

            if self.cbits_ctrl_value > 2 ** len(self.cbits) - 1:
                raise ValueError(
                    "Classical Control value can't be greater than 2^num_cbits -1"
                    f", got {self.cbits_ctrl_value}."
                )

    @property
    def controls(self) -> tuple[int]:
        return self.qubits[: self.operation.num_ctrl_qubits]

    @property
    def targets(self) -> tuple[int]:
        return self.qubits[self.operation.num_ctrl_qubits :]

    def is_gate_instruction(self) -> bool:
        return True

    def to_qasm(self, qasm_out):
        gate = self.operation
        args = None
        if isinstance(gate, ParametrizedGate):
            args = gate.arg_value

        qasm_gate = qasm_out.qasm_name(gate.name)
        if not qasm_gate:
            error_str = f"{self.name} gate's qasm defn is not specified"
            raise NotImplementedError(error_str)

        if self.cbits:
            err_msg = "Exporting controlled gates is not implemented yet."
            raise NotImplementedError(err_msg)
        else:
            qasm_out.output(
                qasm_out._qasm_str(
                    q_name=qasm_gate,
                    q_targets=list(self.targets),
                    q_controls=list(self.controls),
                    q_args=args,
                )
            )

    def __str__(self):
        return f"Gate({self.operation}), qubits({self.qubits}),\
                cbits({self.cbits}), style({self.style})"


@dataclass(frozen=True)
class MeasurementInstruction(CircuitInstruction):
    operation: Measurement

    def __post_init__(self):
        super().__post_init__()
        if not self.is_measurement_instruction():
            raise ValueError(
                f"Operation must be a measurement, got {self.operation}"
            )

        if len(self.qubits) != len(self.cbits):
            raise ValueError(
                "Measurement requires equal number of qubits and cbits."
            )

    def is_measurement_instruction(self) -> bool:
        return True

    def to_qasm(self, qasm_out):
        qasm_out.output(
            "measure q[{}] -> c[{}]".format(self.qubits[0], self.cbits[0])
        )

    def __str__(self):
        return f"Measure(q{self.qubits} -> c{self.cbits})"
