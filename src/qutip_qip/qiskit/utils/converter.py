"""Conversion of circuits from qiskit to qutip_qip."""

from collections.abc import Iterable
from typing import Type
from qiskit.circuit import QuantumCircuit
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate
import qutip_qip.operations.gates as gates

# TODO Expand this dictionary for all the valid qiskit gates
# https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_library#standard-gates
_map_gates: dict[str, Type[Gate]] = {
    "x": gates.X,
    "y": gates.Y,
    "z": gates.Z,
    "h": gates.H,
    "s": gates.S,
    "sdag": gates.Sdag,
    "t": gates.T,
    "tdag": gates.Tdag,
    "sx": gates.SQRTX,
    "sxdag": gates.SQRTXdag,
    "rx": gates.RX,
    "ry": gates.RY,
    "rz": gates.RZ,
    "p": gates.PHASE,
    "u3": gates.QASMU,
    "swap": gates.SWAP,
}

_map_controlled_gates: dict[str, Type[Gate]] = {
    "cx": gates.CX,
    "cy": gates.CY,
    "cz": gates.CZ,
    "ch": gates.CH,
    "cs": gates.CS,
    "ct": gates.CT,
    "crx": gates.CRX,
    "cry": gates.CRY,
    "crz": gates.CRZ,
    "cp": gates.CPHASE,
    "cu3": gates.CQASMU,
}

_ignore_gates: list[str] = ["id", "barrier"]


def get_qutip_index(bit_index: int | list, total_bits: int) -> int:
    """
    Map the bit index from qiskit to qutip.

    Parameters
    ----------
    bit_index : int
        Qiskit bit index

    total_bits: int
        Total number of bits in qiskit circuit.

    Returns
    -------
    int
        qutip-qip bit index.

    Note
    ----
    When we convert a circuit from qiskit to qutip,
    the 0st bit is mapped to the (n-1)th bit and 1st bit to (n-2)th bit
    and so on. Essentially the bit order is reversed.
    """
    if isinstance(bit_index, Iterable):
        return [get_qutip_index(bit, total_bits) for bit in bit_index]
    else:
        return total_bits - 1 - bit_index


def _get_mapped_bits(bits: list | tuple, bit_map: dict[int, int]) -> list:
    return [bit_map[bit] for bit in bits]


def convert_qiskit_circuit_to_qutip(
    qiskit_circuit: QuantumCircuit,
) -> QubitCircuit:
    """
    Convert a :class:`qiskit.circuit.QuantumCircuit` object
    from ``qiskit`` to ``qutip_qip``'s :class:`.QubitCircuit`.

    Parameters
    ----------
    qiskit_circuit : :class:`qiskit.circuit.QuantumCircuit`
        The :class:`qiskit.circuit.QuantumCircuit` object
        to be converted to :class:`QubitCircuit`.

    Returns
    -------
    :class:`.QubitCircuit`
        The converted circuit in qutip_qip's
        :class:`.QubitCircuit` format.
    """
    qubit_map = {}
    for qiskit_index, qubit in enumerate(qiskit_circuit.qubits):
        qubit_map[qubit] = get_qutip_index(
            qiskit_index, total_bits=qiskit_circuit.num_qubits
        )

    clbit_map = {}
    for qiskit_index, clbit in enumerate(qiskit_circuit.clbits):
        clbit_map[clbit] = get_qutip_index(
            qiskit_index, total_bits=qiskit_circuit.num_clbits
        )

    qutip_circuit = QubitCircuit(
        num_qubits=qiskit_circuit.num_qubits,
        num_cbits=qiskit_circuit.num_clbits,
    )

    qutip_circuit.name = qiskit_circuit.name
    for circuit_instruction in qiskit_circuit.data:
        qiskit_instruction = circuit_instruction.operation
        qiskit_qregs = circuit_instruction.qubits
        qiskit_cregs = circuit_instruction.clbits

        # setting the gate argument values according
        # to the required qutip_qip format
        arg_value = None
        if qiskit_instruction.params:
            arg_value = qiskit_instruction.params

        # add the corresponding gate in qutip_qip
        if qiskit_instruction.name in _map_gates.keys():
            gate = _map_gates[qiskit_instruction.name]
            if gate.is_parametric:
                gate = gate(*arg_value)

            qutip_circuit.add_gate(
                gate,
                targets=_get_mapped_bits(qiskit_qregs, bit_map=qubit_map),
            )

        elif qiskit_instruction.name in _map_controlled_gates.keys():
            gate = _map_controlled_gates[qiskit_instruction.name]
            if gate.is_parametric:
                gate = gate(arg_value)

            # FIXME This doesn't work for multicontrolled gates
            qutip_circuit.add_gate(
                gate,
                targets=_get_mapped_bits(qiskit_qregs[1:], bit_map=qubit_map),
                # The 0th bit is the control bit in qiskit by
                # convention, in case of a controlled operation
                controls=_get_mapped_bits(
                    [qiskit_qregs[0]], bit_map=qubit_map
                ),
            )

        elif qiskit_instruction.name == "measure":
            qutip_circuit.add_measurement(
                "measure",
                targets=_get_mapped_bits(qiskit_qregs, bit_map=qubit_map),
                classical_store=clbit_map[qiskit_cregs[0]],
            )

        elif qiskit_instruction.name in _ignore_gates:
            pass

    return qutip_circuit
