# Needed to defer evaluating type hints so that we don't need forward
# references and can hide type hint–only imports from runtime usage.
from __future__ import annotations

from base64 import b64decode
from enum import Enum, auto
from operator import mod
import os
from tempfile import NamedTemporaryFile
from typing import Union, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Literal

try:
    import pyqir.generator as pqg
except ImportError as ex:
    raise ImportError("qutip.qip.qir depends on PyQIR") from ex

try:
    import pyqir.parser as pqp
except ImportError as ex:
    raise ImportError("qutip.qip.qir depends on PyQIR") from ex


from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, Measurement

__all__ = ["circuit_to_qir", "QirFormat"]


class QirFormat(Enum):
    """
    Specifies the format used to serialize QIR.
    """

    #: Specifies that QIR should be encoded as LLVM bitcode (typically, files
    #: ending in `.bc`).
    BITCODE = auto()
    #: Specifies that QIR should be encoded as plain text (typicaly, files
    #: ending in `.ll`).
    TEXT = auto()
    #: Specifies that QIR should be encoded as a PyQIR module object.
    MODULE = auto()

    @classmethod
    def ensure(
        cls, val: Union[Literal["bitcode", "text", "module"], QirFormat]
    ) -> QirFormat:
        """
        Given a value, returns a value ensured to be of type `QirFormat`,
        attempting to convert if needed.
        """
        if isinstance(val, cls):
            return val
        elif isinstance(val, str):
            return cls[val.upper()]

        return cls(val)


# Specify return types for each different format, so that IDE tooling and type
# checkers can resolve the return type based on arguments.
@overload
def circuit_to_qir(
    circuit: QubitCircuit,
    format: Union[Literal[QirFormat.BITCODE], Literal["bitcode"]],
    module_name: str,
) -> bytes: ...


@overload
def circuit_to_qir(
    circuit: QubitCircuit,
    format: Union[Literal[QirFormat.TEXT], Literal["text"]],
    module_name: str,
) -> str: ...


@overload
def circuit_to_qir(
    circuit: QubitCircuit,
    format: Union[Literal[QirFormat.MODULE], Literal["module"]],
    module_name: str,
) -> pqp.QirModule: ...


def circuit_to_qir(circuit, format, module_name="qutip_circuit"):
    """Converts a qubit circuit to its representation in QIR.

    Given a circuit acting on qubits, generates a representation of that
    circuit using Quantum Intermediate Representation (QIR).

    Parameters
    ----------
    circuit
        The circuit to be translated to QIR.
    format
        The QIR serialization to be used. If `"text"`, returns a
        plain-text representation using LLVM IR. If `"bitcode"`, returns a
        dense binary representation ideal for use with other compilation tools.
        If `"module"`, returns a PyQIR module object that can be manipulated
        further before generating QIR.
    module_name
        The name of the module to be emitted.

    Returns
    -------
    A QIR representation of `circuit`, encoded using the format specified by
    `format`.
    """

    # Define as an inner function to make it easier to call from conditional
    # branches.
    def append_operation(
        module: pqg.SimpleModule, builder: pqg.BasicQisBuilder, op: Gate
    ):
        if op.classical_controls:
            result = op.classical_controls[0]
            value = "zero" if op.classical_control_value == 0 else "one"
            # Pull off the first control and recurse.
            op_with_less_controls = Gate(**op.__dict__)
            op_with_less_controls.classical_controls = (
                op_with_less_controls.classical_controls[1:]
            )
            op_with_less_controls.classical_control_value = (
                op_with_less_controls.classical_control_value
                if isinstance(
                    op_with_less_controls.classical_control_value, int
                )
                else (
                    (op_with_less_controls.classical_control_value[1:])
                    if op_with_less_controls.classical_control_value
                    is not None
                    else None
                )
            )
            branch_body = {
                value: (
                    lambda: append_operation(
                        module, builder, op_with_less_controls
                    )
                )
            }
            builder.if_result(module.results[result], **branch_body)
            return

        if op.controls:
            if op.name not in ("CNOT", "CX", "CZ") or len(op.controls) != 1:
                raise NotImplementedError(
                    "Arbitrary controlled quantum operations are not yet supported."
                )

        if op.name == "X":
            builder.x(module.qubits[op.targets[0]])
        elif op.name == "Y":
            builder.y(module.qubits[op.targets[0]])
        elif op.name == "Z":
            builder.z(module.qubits[op.targets[0]])
        elif op.name == "S":
            builder.s(module.qubits[op.targets[0]])
        elif op.name == "T":
            builder.t(module.qubits[op.targets[0]])
        elif op.name == "SNOT":
            builder.h(module.qubits[op.targets[0]])
        elif op.name in ("CNOT", "CX"):
            builder.cx(
                module.qubits[op.controls[0]], module.qubits[op.targets[0]]
            )
        elif op.name == "CZ":
            builder.cz(
                module.qubits[op.controls[0]], module.qubits[op.targets[0]]
            )
        elif op.name == "RX":
            builder.rx(op.arg_value, module.qubits[op.targets[0]])
        elif op.name == "RY":
            builder.ry(op.arg_value, module.qubits[op.targets[0]])
        elif op.name == "RZ":
            builder.rz(op.arg_value, module.qubits[op.targets[0]])
        elif op.name in ("CRZ", "TOFFOLI"):
            raise NotImplementedError(
                "Decomposition of CRZ and Toffoli gates into base "
                + "profile instructions is not yet implemented."
            )
        else:
            raise ValueError(
                f"Gate {op.name} not supported by the basic QIR builder, "
                + "and may require a custom declaration."
            )

    fmt = QirFormat.ensure(format)

    module = pqg.SimpleModule(module_name, circuit.N, circuit.num_cbits or 0)
    builder = pqg.BasicQisBuilder(module.builder)

    for op in circuit.gates:
        # If we have a QuTiP gate, then we need to convert it into one of
        # the reserved operation names in the QIR base profile's quantum
        # instruction set (QIS).
        if isinstance(op, Gate):
            # TODO: Validate indices.
            append_operation(module, builder, op)

        elif isinstance(op, Measurement):
            builder.mz(
                module.qubits[op.targets[0]],
                module.results[op.classical_store],
            )

        else:
            raise NotImplementedError(
                f"Instruction {op} is not implemented in the QIR base "
                + "profile and may require a custom declaration."
            )

    if fmt == QirFormat.TEXT:
        return module.ir()
    elif fmt == QirFormat.BITCODE:
        return module.bitcode()
    elif fmt == QirFormat.MODULE:
        bitcode = module.bitcode()
        f = NamedTemporaryFile(suffix=".bc", delete=False)
        try:
            f.write(bitcode)
        finally:
            f.close()
        module = pqp.QirModule(f.name)
        try:
            os.unlink(f.name)
        except:
            pass
        return module
    else:
        assert (
            False
        ), "Internal error; should have caught invalid format enum earlier."
