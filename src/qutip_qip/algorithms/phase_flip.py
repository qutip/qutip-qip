from qutip_qip.circuit import QubitCircuit

__all__ = ["PhaseFlipCode"]


class PhaseFlipCode:
    """
    Implementation of the 3-qubit phase-flip quantum error correction code.

    This code protects against a single Z (phase) error by encoding the logical
    qubit into a 3-qubit entangled state using Hadamard transformations and
    CNOT gates, then measuring parity between pairs of qubits using ancilla qubits.
    Conditional Z corrections are applied based on classical measurement outcomes.
    """

    def __init__(self):
        """
        Initializes the PhaseFlipCode with 3 data qubits and 2 syndrome (ancilla) qubits.
        """
        self._n_data = 3
        self._n_syndrome = 2

    @property
    def n_data(self):
        """
        Returns:
            int: Number of data qubits (3).
        """
        return self._n_data

    @property
    def n_syndrome(self):
        """
        Returns:
            int: Number of syndrome qubits (2).
        """
        return self._n_syndrome

    def encode_circuit(self, data_qubits):
        """
        Constructs the encoding circuit for the phase-flip code.

        The logical qubit is encoded into an entangled state in the X-basis using Hadamard
        (SNOT) gates followed by two CNOT gates. This creates redundancy to detect and correct
        a single phase error.

        Args:
            data_qubits (list[int]): Indices of 3 data qubits.

        Returns:
            QubitCircuit: The encoding quantum circuit.

        Raises:
            ValueError: If the number of data qubits is not 3.
        """
        if len(data_qubits) != 3:
            raise ValueError("Expected 3 data qubits.")
        qc = QubitCircuit(max(data_qubits) + 1)

        # Convert to X-basis
        for q in data_qubits:
            qc.add_gate("SNOT", targets=[q])

        # Bit-flip-style encoding
        control = data_qubits[0]
        for target in data_qubits[1:]:
            qc.add_gate("CNOT", controls=control, targets=target)

        return qc

    def syndrome_and_correction_circuit(self, data_qubits, syndrome_qubits):
        """
        Builds the circuit for syndrome extraction and correction.

        Parity is measured between data qubit pairs using ancillas and CNOT gates.
        Measurements are stored in classical bits, and Z corrections are applied
        conditionally based on the measured syndrome.

        Args:
            data_qubits (list[int]): Indices of 3 data qubits.
            syndrome_qubits (list[int]): Indices of 2 syndrome qubits.

        Returns:
            QubitCircuit: Circuit for syndrome measurement and Z correction.

        Raises:
            ValueError: If the number of qubits is incorrect.
        """
        if len(data_qubits) != 3 or len(syndrome_qubits) != 2:
            raise ValueError("Expected 3 data qubits and 2 syndrome qubits.")

        total_qubits = max(data_qubits + syndrome_qubits) + 1
        qc = QubitCircuit(N=total_qubits, num_cbits=2)

        dq = data_qubits
        sq = syndrome_qubits

        # Parity checks
        qc.add_gate("CNOT", controls=dq[0], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[0])
        qc.add_gate("CNOT", controls=dq[1], targets=sq[1])
        qc.add_gate("CNOT", controls=dq[2], targets=sq[1])

        # Measure syndrome qubits
        qc.add_measurement(sq[0], sq[0], classical_store=0)
        qc.add_measurement(sq[1], sq[1], classical_store=1)

        # Classically controlled Z corrections
        qc.add_gate(
            "Z",
            targets=dq[0],
            classical_controls=[0, 1],
            classical_control_value=2,
        )
        qc.add_gate(
            "Z",
            targets=dq[1],
            classical_controls=[0, 1],
            classical_control_value=3,
        )
        qc.add_gate(
            "Z",
            targets=dq[2],
            classical_controls=[0, 1],
            classical_control_value=1,
        )

        return qc

    def decode_circuit(self, data_qubits):
        """
        Constructs the decoding circuit that reverses the encoding operation.

        It first applies the inverse of the CNOT encoding, then converts the qubits
        back from the X-basis to the Z-basis using Hadamard (SNOT) gates.

        Args:
            data_qubits (list[int]): Indices of 3 data qubits.

        Returns:
            QubitCircuit: The decoding circuit.

        Raises:
            ValueError: If the number of data qubits is not 3.
        """
        if len(data_qubits) != 3:
            raise ValueError("Expected 3 data qubits.")
        qc = QubitCircuit(max(data_qubits) + 1)

        control = data_qubits[0]
        for target in reversed(data_qubits[1:]):
            qc.add_gate("CNOT", controls=control, targets=target)

        # Convert back from X-basis
        for q in data_qubits:
            qc.add_gate("SNOT", targets=[q])

        return qc
