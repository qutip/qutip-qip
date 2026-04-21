from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import CX, H, Z, TOFFOLI
from qutip_qip.operations.measurement import Measurement
from qutip_qip.typing import IntSequence


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
    def n_data(self) -> int:
        """
        Returns
        -------
        int
            Number of data qubits (3).
        """
        return self._n_data

    @property
    def n_syndrome(self) -> int:
        """
        Returns
        -------
        int
            Number of syndrome qubits (2).
        """
        return self._n_syndrome

    def encode_circuit(self, data_qubits: IntSequence) -> QubitCircuit:
        """
        Constructs the encoding circuit for the phase-flip code.

        The logical qubit is first encoded by two CX gates and then converted to the X-basis
        using Hadamard (H). This creates redundancy to detect and correct a single phase error.

        Parameters
        ----------
        data_qubits : sequence of int
            Indices of 3 data qubits.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            The encoding quantum circuit.

        Raises
        ------
        ValueError: If the number of data qubits is not 3.
        """
        if len(data_qubits) != 3:
            raise ValueError("Expected 3 data qubits.")

        qc = QubitCircuit(max(data_qubits) + 1)

        # Bit-flip-style encoding
        control = data_qubits[0]
        for target in data_qubits[1:]:
            qc.add_gate(CX, controls=control, targets=target)

        # Convert to X-basis
        for q in data_qubits:
            qc.add_gate(H, targets=[q])

        return qc

    def syndrome_and_correction_circuit(
        self, data_qubits: IntSequence, syndrome_qubits: IntSequence
    ) -> QubitCircuit:
        """
        Builds the circuit for syndrome extraction and correction.

        The data qubits are temporarily converted back to the Z-basis so parity
        can be measured between pairs using ancillas and CNOT gates.
        Measurements are stored in classical bits, and Z corrections are applied
        conditionally based on the measured syndrome.

        Parameters
        ----------
        data_qubits : sequence of int
            Indices of 3 data qubits.
        syndrome_qubits : sequence of int
            Indices of 2 syndrome qubits.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            Circuit for syndrome measurement and Z correction.

        Raises
        ------
        ValueError: If there are not exactly 3 data qubits and 2 syndrome qubits.
        """
        if len(data_qubits) != 3 or len(syndrome_qubits) != 2:
            raise ValueError("Expected 3 data qubits and 2 syndrome qubits.")

        total_qubits = max(data_qubits + syndrome_qubits) + 1
        qc = QubitCircuit(num_qubits=total_qubits, num_cbits=2)

        dq = data_qubits
        sq = syndrome_qubits

        # Convert back from X-basis
        for q in data_qubits:
            qc.add_gate(H, targets=[q])

        # Parity checks
        qc.add_gate(CX, controls=dq[0], targets=sq[0])
        qc.add_gate(CX, controls=dq[1], targets=sq[0])
        qc.add_gate(CX, controls=dq[1], targets=sq[1])
        qc.add_gate(CX, controls=dq[2], targets=sq[1])

        # Convert to X-basis
        for q in data_qubits:
            qc.add_gate(H, targets=[q])

        # Measure syndrome qubits
        qc.add_measurement(Measurement(), sq[0], classical_store=0)
        qc.add_measurement(Measurement(), sq[1], classical_store=1)

        # Classically controlled Z corrections
        qc.add_gate(
            Z,
            targets=dq[0],
            classical_controls=[0, 1],
            classical_control_value=0b10,
        )
        qc.add_gate(
            Z,
            targets=dq[1],
            classical_controls=[0, 1],
            classical_control_value=0b11,
        )
        qc.add_gate(
            Z,
            targets=dq[2],
            classical_controls=[0, 1],
            classical_control_value=0b01,
        )

        return qc

    def decode_circuit(self, data_qubits: IntSequence) -> QubitCircuit:
        """
        Constructs the decoding circuit that reverses the encoding operation.

        It first applies the Hadamard (H) gates to convert the qubits back from
        the X-basis to the Z-basis, then applies the inverse of the CX encoding to
        decode the qubits. A Toffoli gate is applied in the end to verify parity.

        Parameters
        ----------
        data_qubits : sequence of int
            Indices of 3 data qubits.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            The decoding circuit.

        Raises
        ------
        ValueError: If the number of data qubits is not 3.
        """
        if len(data_qubits) != 3:
            raise ValueError("Expected 3 data qubits.")
        qc = QubitCircuit(max(data_qubits) + 1)

        # Convert back from X-basis
        for q in data_qubits:
            qc.add_gate(H, targets=[q])

        control = data_qubits[0]
        for target in reversed(data_qubits[1:]):
            qc.add_gate(CX, controls=control, targets=target)

        qc.add_gate(TOFFOLI, controls=data_qubits[1:], targets=control)

        return qc
