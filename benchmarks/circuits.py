from qutip_qip.circuit import QubitCircuit
import numpy as np


def ghz_circuit(n_qubits: int) -> QubitCircuit:
    qc = QubitCircuit(n_qubits)
    qc.add_gate("SNOT", targets=[0])
    for i in range(n_qubits - 1):
        qc.add_gate("CNOT", controls=[i], targets=[i + 1])
    return qc


def random_layered_circuit(n_qubits: int, depth: int = 5) -> QubitCircuit:
    qc = QubitCircuit(n_qubits)
    rng = np.random.default_rng(42)

    for _ in range(depth):
        for q in range(n_qubits):
            qc.add_gate("RX", targets=[q], arg_value=float(rng.random()))
        for q in range(n_qubits - 1):
            qc.add_gate("CNOT", controls=[q], targets=[q + 1])
    return qc

