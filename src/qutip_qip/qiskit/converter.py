from qutip_qip.circuit import QubitCircuit
from qiskit.quantum_info import Operator
import numpy as np
from qutip import Qobj

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


def _make_user_gate(unitary, inst):
    """
    Returns a user defined gate from a unitary matrix.

    Parameters
    ----------
    unitary : numpy.array
        The unitary matrix describing the gate.

    inst : qiskit.circuit.Instruction
        Qiskit instruction containing info about the gate.
    """

    def user_gate():
        return Qobj(
            unitary, dims=[[2] * inst.num_qubits, [2] * inst.num_qubits]
        )

    return user_gate


def qiskit_to_qutip(qiskit_circ):
    """
    Convert a QuantumCircuit from qiskit to qutip_qip's QubitCircuit.

    Parameters
    ----------
    qiskit_circ : QuantumCircuit
        QuantumCircuit to be converted to QubitCircuit. 
    """
    qubit_map = {}
    for i, qubit in enumerate(qiskit_circ.qubits):
        qubit_map[qubit] = i

    clbit_map = {}
    for i, clbit in enumerate(qiskit_circ.clbits):
        clbit_map[clbit] = i

    qutip_circ = QubitCircuit(
        N=qiskit_circ.num_qubits, num_cbits=qiskit_circ.num_clbits
    )

    for gate in qiskit_circ.data:
        inst = gate[0]
        inst_name = inst.name
        qregs = gate[1]
        cregs = gate[2]

        # setting the argument values according
        # to the required qutip_qip format
        arg_value = None if not inst.params else inst.params
        if not inst.params:
            arg_value = None
        elif len(inst.params) == 1:
            arg_value = inst.params[0]
        else:
            arg_value = inst.params

        # add the corresponding gate in qutip_qip
        if inst_name in _map_gates.keys():
            qutip_circ.add_gate(
                _map_gates[inst_name],
                targets=[qubit_map[qreg] for qreg in qregs],
                arg_value=arg_value,
            )
        elif inst_name in _map_controlled_gates.keys():
            qutip_circ.add_gate(
                _map_controlled_gates[inst_name],
                controls=[qubit_map[qregs[0]]],
                targets=[qubit_map[qreg]
                         for qreg in qregs[1:]],
                arg_value=arg_value,
            )
        elif inst_name == "measure":
            qutip_circ.add_measurement("measure", targets=[
                                       qubit_map[qreg] for qreg in qregs], classical_store=clbit_map[cregs[0]])

        elif inst_name in _ignore_gates:
            pass
        else:
            unitary = np.array(Operator(inst))
            qutip_circ.user_gates[inst_name] = _make_user_gate(unitary, inst)
            qutip_circ.add_gate(
                inst_name, targets=[qubit_map[qreg] for qreg in qregs]
            )

    return qutip_circ
