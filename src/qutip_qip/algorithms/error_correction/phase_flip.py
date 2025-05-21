from qutip_qip.circuit import QubitCircuit


class PhaseFlipCode:
    """
    Generalized implementation of the 3-qubit phase-flip code.
    Protects against phase-flip (Z) errors using logical encoding.

    Parameters
    ----------
    data_qubits : list of int
        The three physical qubits holding the logical qubit.
    syndrome_qubits : list of int
        Two ancilla qubits used for syndrome extraction.
    """

    def __init__(self, data_qubits=[0, 1, 2], syndrome_qubits=[3, 4]):
        assert len(data_qubits) == 3, "Phase-flip code requires 3 data qubits."
        self.data_qubits = data_qubits
        self.syndrome_qubits = syndrome_qubits
        self.n_qubits = max(data_qubits + syndrome_qubits) + 1

    def encode_circuit(self):
        qc = QubitCircuit(self.n_qubits)
        # Convert to |+⟩ and then entangle
        for q in self.data_qubits:
            qc.add_gate("SNOT", targets=q)  # Hadamard

        control = self.data_qubits[0]
        for target in self.data_qubits[1:]:
            qc.add_gate("CNOT", controls=control, targets=target)
        return qc

    def syndrome_measurement_circuit(self):
        qc = QubitCircuit(self.n_qubits)
        dq = self.data_qubits
        sq = self.syndrome_qubits

        # Hadamard before parity check to convert Z errors into X
        for q in dq:
            qc.add_gate("SNOT", targets=q)

        qc.add_gate("CNOT", controls=dq[0], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[1])
        qc.add_gate("CNOT", controls=dq[2], targets=sq[1])

        # Hadamard after parity check to restore basis
        for q in dq:
            qc.add_gate("SNOT", targets=q)

        return qc

    def correction_circuit(self, syndrome):
        qc = QubitCircuit(self.n_qubits)
        s1, s2 = syndrome
        if s1 == 1 and s2 == 0:
            qc.add_gate("Z", targets=self.data_qubits[0])
        elif s1 == 1 and s2 == 1:
            qc.add_gate("Z", targets=self.data_qubits[1])
        elif s1 == 0 and s2 == 1:
            qc.add_gate("Z", targets=self.data_qubits[2])
        return qc

    def decode_circuit(self):
        qc = QubitCircuit(self.n_qubits)
        control = self.data_qubits[0]
        for target in reversed(self.data_qubits[1:]):
            qc.add_gate("CNOT", controls=control, targets=target)
        qc.add_gate("TOFFOLI", controls=self.data_qubits[1:], targets=control)
        for q in self.data_qubits:
            qc.add_gate("SNOT", targets=q)
        return qc
