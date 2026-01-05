"""Conversion of circuits from qiskit to qutip_qip."""

from qiskit.circuit import QuantumCircuit
from qutip_qip.circuit import QubitCircuit


_map_gates: dict[str, str] = {
    "p": "PHASEGATE",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "SNOT",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "swap": "SWAP",
    "u": "QASMU",
}

_map_controlled_gates: dict[str, str] = {
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "cp": "CPHASE",
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

    if isinstance(bit_index, list):
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
        N=qiskit_circuit.num_qubits, num_cbits=qiskit_circuit.num_clbits
    )

    qutip_circuit.name = qiskit_circuit.name

    for circuit_instruction in qiskit_circuit.data:
        qiskit_instruction = circuit_instruction.operation
        qiskit_qregs = circuit_instruction.qubits
        qiskit_cregs = circuit_instruction.clbits

        # setting the gate argument values according
        # to the required qutip_qip format
        if not qiskit_instruction.params:
            arg_value = None
        elif len(qiskit_instruction.params) == 1:
            arg_value = qiskit_instruction.params[0]
        else:
            arg_value = qiskit_instruction.params

        # add the corresponding gate in qutip_qip
        if qiskit_instruction.name in _map_gates.keys():
            qutip_circuit.add_gate(
                _map_gates[qiskit_instruction.name],
                targets=_get_mapped_bits(qiskit_qregs, bit_map=qubit_map),
                arg_value=arg_value,
            )

        elif qiskit_instruction.name in _map_controlled_gates.keys():
            qutip_circuit.add_gate(
                _map_controlled_gates[qiskit_instruction.name],
                # The 0th bit is the control bit in qiskit by
                # convention, in case of a controlled operation
                controls=_get_mapped_bits(
                    [qiskit_qregs[0]], bit_map=qubit_map
                ),
                targets=_get_mapped_bits(qiskit_qregs[1:], bit_map=qubit_map),
                arg_value=arg_value,
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
