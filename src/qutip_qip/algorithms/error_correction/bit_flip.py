from qutip_qip.circuit import QubitCircuit

class BitFlipCode:
    """
    Generalized implementation of the 3-qubit bit-flip code.

    Parameters
    ----------
    data_qubits : list of int
        The three physical qubits holding the logical qubit.
    syndrome_qubits : list of int, optional
        Two ancilla qubits used for syndrome extraction.
    """

    def __init__(self, data_qubits=[0, 1, 2], syndrome_qubits=[3, 4]):
        assert len(data_qubits) == 3, "Bit-flip code requires 3 data qubits."
        assert len(syndrome_qubits) == 2, "Syndrome extraction requires 2 ancilla qubits."
        self.data_qubits = data_qubits
        self.syndrome_qubits = syndrome_qubits
        self.n_qubits = max(data_qubits + syndrome_qubits) + 1

    def encode_circuit(self):
        """
        Returns
        -------
        QubitCircuit
            Circuit encoding the logical qubit into 3 physical qubits.
        """
        qc = QubitCircuit(self.n_qubits)
        control = self.data_qubits[0]
        for target in self.data_qubits[1:]:
            qc.add_gate("CNOT", controls=control, targets=target)
        return qc

    def syndrome_measurement_circuit(self):
        """
        Returns
        -------
        QubitCircuit
            Circuit to extract the error syndrome using ancilla qubits.
        """
        qc = QubitCircuit(self.n_qubits)
        dq = self.data_qubits
        sq = self.syndrome_qubits

        # First syndrome bit: parity of dq[0] and dq[1]
        qc.add_gate("CNOT", controls=dq[0], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[0])

        # Second syndrome bit: parity of dq[1] and dq[2]
        qc.add_gate("CNOT", controls=dq[1], targets=sq[1])
        qc.add_gate("CNOT", controls=dq[2], targets=sq[1])

        return qc

    def correction_circuit(self, syndrome):
        """
        Parameters
        ----------
        syndrome : tuple
            Two-bit syndrome measurement result (s1, s2).

        Returns
        -------
        QubitCircuit
            Circuit applying the appropriate X gate based on syndrome.
        """
        qc = QubitCircuit(self.n_qubits)
        s1, s2 = syndrome

        if s1 == 1 and s2 == 0:
            qc.add_gate("X", targets=self.data_qubits[0])
        elif s1 == 1 and s2 == 1:
            qc.add_gate("X", targets=self.data_qubits[1])
        elif s1 == 0 and s2 == 1:
            qc.add_gate("X", targets=self.data_qubits[2])
        # No correction for (0,0)
        return qc

    def decode_circuit(self):
        """
        Returns
        -------
        QubitCircuit
            Circuit to decode the logical qubit back to original qubit.
        """
        qc = QubitCircuit(self.n_qubits)
        control = self.data_qubits[0]
        for target in reversed(self.data_qubits[1:]):
            qc.add_gate("CNOT", controls=control, targets=target)

        # Optional TOFFOLI to verify parity
        qc.add_gate("TOFFOLI", controls=self.data_qubits[1:], targets=control)
        return qc
