"""Conversion of circuits from qiskit to qutip_qip."""

from qutip_qip.circuit import QubitCircuit
from qiskit.quantum_info import Operator
import numpy as np
from qutip import Qobj
from typing import Union
import qiskit
from qiskit.circuit import QuantumCircuit

_map_gates = {
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

_map_controlled_gates = {
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "cp": "CPHASE",
}

_ignore_gates = ["id", "barrier"]


def _make_user_gate(
    unitary: np.ndarray, qiskit_instruction: qiskit.circuit.Instruction
):
    """
    Returns a user defined gate from a unitary matrix.

    Parameters
    ----------
    unitary : numpy.ndarray
        The unitary matrix describing the gate.

    qiskit_instruction : qiskit.circuit.Instruction
        Qiskit instruction containing info about the gate.
    """

    def user_gate():
        return Qobj(
            unitary,
            dims=[
                [2] * qiskit_instruction.num_qubits,
                [2] * qiskit_instruction.num_qubits,
            ],
        )

    return user_gate


def _get_qutip_index(bit_index: Union[int, list], total_bits: int) -> int:
    """
    Map the bit index from qiskit to qutip.

    Note
    ----
    When we convert a circuit from qiskit to qutip,
    the 0st bit is mapped to the (n-1)th bit and so on.


    """

    if isinstance(bit_index, list):
        return [_get_qutip_index(bit, total_bits) for bit in bit_index]
    else:
        return total_bits - 1 - bit_index


def _get_mapped_bits(bits: Union[list, tuple], bit_map: dict) -> list:
    return [bit_map[bit] for bit in bits]


def convert_qiskit_circuit(
    qiskit_circuit: QuantumCircuit, allow_custom_gate=True
) -> QubitCircuit:
    """
    Convert a :class:`qiskit.circuit.QuantumCircuit` object
    from ``qiskit`` to ``qutip_qip``'s :class:`.QubitCircuit`.

    Parameters
    ----------
    qiskit_circuit : :class:`qiskit.circuit.QuantumCircuit`
        The :class:`qiskit.circuit.QuantumCircuit` object
        to be converted to :class:`QubitCircuit`.

    allow_custom_gate : bool
        If False, this function will raise an error if
        gate conversion is done using a custom gate's
        unitary matrix.

    Returns
    -------
    :class:`.QubitCircuit`
        The converted circuit in qutip_qip's
        :class:`.QubitCircuit` format.
    """
    qubit_map = {}
    for qiskit_index, qubit in enumerate(qiskit_circuit.qubits):
        qubit_map[qubit] = _get_qutip_index(
            qiskit_index, total_bits=qiskit_circuit.num_qubits
        )

    clbit_map = {}
    for qiskit_index, clbit in enumerate(qiskit_circuit.clbits):
        clbit_map[clbit] = _get_qutip_index(
            qiskit_index, total_bits=qiskit_circuit.num_clbits
        )

    qutip_circuit = QubitCircuit(
        N=qiskit_circuit.num_qubits, num_cbits=qiskit_circuit.num_clbits
    )

    qutip_circuit.name = qiskit_circuit.name

    for qiskit_gate in qiskit_circuit.data:
        # qiskit_gate stores info about the gate, target
        # qubits and classical bits (for measurements)
        qiskit_instruction = qiskit_gate[0]
        qiskit_qregs = qiskit_gate[1]
        qiskit_cregs = qiskit_gate[2]

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

        else:
            if not allow_custom_gate:
                raise RuntimeError(
                    f"{qiskit_instruction.name} is not a \
gate in qutip_qip. To simulate this gate using it's corresponding \
unitary matrix, set allow_custom_gate=True."
                )

            unitary = np.array(Operator(qiskit_instruction))
            qutip_circuit.user_gates[qiskit_instruction.name] = (
                _make_user_gate(unitary, qiskit_instruction)
            )
            qutip_circuit.add_gate(
                qiskit_instruction.name,
                targets=_get_mapped_bits(qiskit_qregs, bit_map=qubit_map),
            )

    return qutip_circuit
