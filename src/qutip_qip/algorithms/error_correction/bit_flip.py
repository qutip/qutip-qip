from qutip_qip.circuit import QubitCircuit

class BitFlipCode:
    """
    Implementation of the 3-qubit bit-flip code using projective measurements
    and classically controlled gates for automatic correction.
    """

    def __init__(self):
        self._n_data = 3
        self._n_syndrome = 2

    @property
    def n_data(self):
        return self._n_data

    @property
    def n_syndrome(self):
        return self._n_syndrome

    def encode_circuit(self, data_qubits):
        if len(data_qubits) != self.n_data:
            raise ValueError(f"Expected {self.n_data} data qubits, got {len(data_qubits)}.")
        qc = QubitCircuit(max(data_qubits) + 1)
        control = data_qubits[0]
        for target in data_qubits[1:]:
            qc.add_gate("CNOT", controls=control, targets=target)
        return qc

    def syndrome_and_correction_circuit(self, data_qubits, syndrome_qubits):
        if len(data_qubits) != self.n_data:
            raise ValueError(f"Expected {self.n_data} data qubits, got {len(data_qubits)}.")
        if len(syndrome_qubits) != self.n_syndrome:
            raise ValueError(f"Expected {self.n_syndrome} syndrome qubits, got {len(syndrome_qubits)}.")

        total_qubits = max(data_qubits + syndrome_qubits) + 1
        classical_bits = len(syndrome_qubits)
        qc = QubitCircuit(N=total_qubits, num_cbits=classical_bits)

        dq = data_qubits
        sq = syndrome_qubits

        # Syndrome extraction
        qc.add_gate("CNOT", controls=dq[0], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[1])
        qc.add_gate("CNOT", controls=dq[2], targets=sq[1])

        # Measurements into classical bits
        qc.add_measurement(sq[0], sq[0], classical_store=0)
        qc.add_measurement(sq[1], sq[1], classical_store=1)

        # Classically controlled correction
        qc.add_gate("X", targets=dq[0], classical_controls=[0, 1], classical_control_value=2)
        qc.add_gate("X", targets=dq[1], classical_controls=[0, 1], classical_control_value=3)
        qc.add_gate("X", targets=dq[2], classical_controls=[0, 1], classical_control_value=1)

        return qc

    def decode_circuit(self, data_qubits):
        if len(data_qubits) != self.n_data:
            raise ValueError(f"Expected {self.n_data} data qubits, got {len(data_qubits)}.")
        qc = QubitCircuit(max(data_qubits) + 1)
        control = data_qubits[0]
        for target in reversed(data_qubits[1:]):
            qc.add_gate("CNOT", controls=control, targets=target)

        # Optional parity verification
        qc.add_gate("TOFFOLI", controls=data_qubits[1:], targets=control)
        return qc

