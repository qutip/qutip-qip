from qutip_qip.circuit import QubitCircuit

class BitFlipCode:
    """
    Implementation of the 3-qubit bit-flip code using projective measurements
    and classically controlled gates for automatic correction.

    The class represents the abstract structure of the bit-flip code.
    Qubit indices must be provided when generating the circuit.

    Methods
    -------
    encode_circuit(data_qubits)
        Encode a logical qubit into three physical qubits.
    syndrome_and_correction_circuit(data_qubits, syndrome_qubits)
        Extract error syndrome and apply correction via classical control.
    decode_circuit(data_qubits)
        Decode the logical qubit back to a single physical qubit.
    """

    def __init__(self):
        self.n_data = 3
        self.n_syndrome = 2

    def encode_circuit(self, data_qubits):
        assert len(data_qubits) == self.n_data
        qc = QubitCircuit(max(data_qubits) + 1)
        control = data_qubits[0]
        for target in data_qubits[1:]:
            qc.add_gate("CNOT", controls=control, targets=target)
        return qc

    def syndrome_and_correction_circuit(self, data_qubits, syndrome_qubits):
        assert len(data_qubits) == self.n_data
        assert len(syndrome_qubits) == self.n_syndrome

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
        qc.add_measurement(sq[0], classical_store=0)
        qc.add_measurement(sq[1], classical_store=1)

        # Classically controlled correction
        qc.add_gate("X", targets=dq[0], classical_controls=[0, 1], classical_control_value=[1, 0])
        qc.add_gate("X", targets=dq[1], classical_controls=[0, 1], classical_control_value=[1, 1])
        qc.add_gate("X", targets=dq[2], classical_controls=[0, 1], classical_control_value=[0, 1])

        return qc

    def decode_circuit(self, data_qubits):
        assert len(data_qubits) == self.n_data
        qc = QubitCircuit(max(data_qubits) + 1)
        control = data_qubits[0]
        for target in reversed(data_qubits[1:]):
            qc.add_gate("CNOT", controls=control, targets=target)

        # Optional parity verification
        qc.add_gate("TOFFOLI", controls=data_qubits[1:], targets=control)
        return qc
