from qutip_qip.circuit import QubitCircuit
from qiskit.quantum_info import Operator
import numpy as np
from qutip import Qobj
import warnings

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


def _make_user_gate(unitary, instruction):
    """
    Returns a user defined gate from a unitary matrix.

    Parameters
    ----------
    unitary : numpy.array
        The unitary matrix describing the gate.

    instruction : qiskit.circuit.Instruction
        Qiskit instruction containing info about the gate.
    """

    def user_gate():
        return Qobj(
            unitary,
            dims=[[2] * instruction.num_qubits,
                  [2] * instruction.num_qubits]
        )

    return user_gate


def _get_qutip_index(qiskit_index, total_qubits):
    """
    Map the bit index from qiskit to qutip.

    When we convert a circuit from qiskit to qutip,
    the 1st bit is mapped to the nth bit and so on.
    """
    return total_qubits - 1 - qiskit_index


def qiskit_to_qutip(qiskit_circuit):
    """
    Convert a QuantumCircuit from qiskit to qutip_qip's QubitCircuit.

    Parameters
    ----------
    qiskit_circuit : QuantumCircuit
        QuantumCircuit to be converted to QubitCircuit. 

    Returns
    -------
    qutip_circuit : QubitCircuit
        The converted circuit in qutip_qip's QubitCircuit format.
    """
    qubit_map = {}
    for qiskit_index, qubit in enumerate(qiskit_circuit.qubits):
        qubit_map[qubit] = _get_qutip_index(
            qiskit_index, qiskit_circuit.num_qubits)

    clbit_map = {}
    for qiskit_index, clbit in enumerate(qiskit_circuit.clbits):
        clbit_map[clbit] = _get_qutip_index(
            qiskit_index, qiskit_circuit.num_qubits)

    qutip_circuit = QubitCircuit(
        N=qiskit_circuit.num_qubits, num_cbits=qiskit_circuit.num_clbits
    )

    for qiskit_gate in qiskit_circuit.data:
        # gate stores info about the gate, target qubits
        # and classical bits (for measurements)
        qiskit_instruction = qiskit_gate[0]
        qiskit_qregs = qiskit_gate[1]
        qiskit_cregs = qiskit_gate[2]

        # setting the gate argument values according
        # to the required qutip_qip format
        arg_value = None if not qiskit_instruction.params else qiskit_instruction.params
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
                targets=[qubit_map[qreg] for qreg in qiskit_qregs],
                arg_value=arg_value,
            )

        elif qiskit_instruction.name in _map_controlled_gates.keys():
            qutip_circuit.add_gate(
                _map_controlled_gates[qiskit_instruction.name],
                controls=[qubit_map[qiskit_qregs[0]]],
                targets=[qubit_map[qreg]
                         for qreg in qiskit_qregs[1:]],
                arg_value=arg_value,
            )

        elif qiskit_instruction.name == "measure":
            qutip_circuit.add_measurement(
                "measure",
                targets=[qubit_map[qreg] for qreg in qiskit_qregs],
                classical_store=clbit_map[qiskit_cregs[0]]
            )

        elif qiskit_instruction.name in _ignore_gates:
            pass

        else:
            warnings.warn(
                f"{qiskit_instruction.name} is not a gate in qutip_qip. \
This gate will be simulated using it's corresponding unitary matrix."
            )

            unitary = np.array(Operator(qiskit_instruction))
            qutip_circuit.user_gates[qiskit_instruction.name] = _make_user_gate(
                unitary, qiskit_instruction)
            qutip_circuit.add_gate(
                qiskit_instruction.name, targets=[
                    qubit_map[qreg] for qreg in qiskit_qregs]
            )

    return qutip_circuit
