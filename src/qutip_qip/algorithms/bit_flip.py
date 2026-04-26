from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import CX, TOFFOLI, X
from qutip_qip.operations.measurement import Mz
from qutip_qip.typing import IntSequence


class BitFlipCode:
    """
    Implementation of the 3-qubit bit-flip quantum error correction code using
    projective measurements and classically controlled correction gates.

    This code detects and corrects a single bit-flip (X) error on any of the three qubits
    using two syndrome (ancilla) qubits.
    """

    def __init__(self):
        """
        Initializes the bit-flip code with 3 data qubits and 2 syndrome qubits.
        """
        self._n_data = 3
        self._n_syndrome = 2

    @property
    def n_data(self) -> int:
        """
        Returns
        -------
        int
            Number of data qubits (always 3 for bit-flip code).
        """
        return self._n_data

    @property
    def n_syndrome(self) -> int:
        """
        Returns
        -------
        int
            Number of syndrome qubits used for error detection (2 for this code).
        """
        return self._n_syndrome

    def encode_circuit(self, data_qubits: IntSequence) -> QubitCircuit:
        """
        Constructs the encoding circuit for the bit-flip code. The first qubit is the control,
        and CNOT gates are applied from it to the other data qubits to encode logical states
        :math:`|0\\rangle` or :math:`|1\\rangle`.

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

        if len(data_qubits) != self.n_data:
            raise ValueError(
                f"Expected {self.n_data} data qubits, got {len(data_qubits)}."
            )
        qc = QubitCircuit(max(data_qubits) + 1)
        control = data_qubits[0]
        for target in data_qubits[1:]:
            qc.add_gate(CX, controls=control, targets=target)
        return qc

    def syndrome_and_correction_circuit(
        self, data_qubits: IntSequence, syndrome_qubits: IntSequence
    ) -> QubitCircuit:
        """
        Constructs the circuit for syndrome extraction and classical error correction.
        The circuit measures parity between qubit pairs and applies X gates conditionally.

        Parameters
        ----------
        data_qubits : sequence of int
            Indices of 3 data qubits.
        syndrome_qubits : sequence of int
            Indices of 2 syndrome qubits.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            Circuit for syndrome measurement and X correction.

        Raises
        ------
        ValueError: If the number of data or syndrome qubits is incorrect.
        """
        if len(data_qubits) != self.n_data:
            raise ValueError(
                f"Expected {self.n_data} data qubits, got {len(data_qubits)}."
            )
        if len(syndrome_qubits) != self.n_syndrome:
            raise ValueError(
                f"Expected {self.n_syndrome} syndrome qubits, got {len(syndrome_qubits)}."
            )

        total_qubits = max(data_qubits + syndrome_qubits) + 1
        classical_bits = len(syndrome_qubits)
        qc = QubitCircuit(num_qubits=total_qubits, num_cbits=classical_bits)

        dq = data_qubits
        sq = syndrome_qubits

        # Syndrome extraction: parity checks
        qc.add_gate(CX, controls=dq[0], targets=sq[0])
        qc.add_gate(CX, controls=dq[1], targets=sq[0])
        qc.add_gate(CX, controls=dq[1], targets=sq[1])
        qc.add_gate(CX, controls=dq[2], targets=sq[1])

        # Measurements into classical registers
        qc.add_measurement(Mz, sq[0], classical_store=0)
        qc.add_measurement(Mz, sq[1], classical_store=1)

        # Classical-controlled corrections based on measurement outcomes
        # 2 (10): X on qubit 0, 3 (11): X on qubit 1, 1 (01): X on qubit 2
        qc.add_gate(
            X,
            targets=dq[0],
            classical_controls=[0, 1],
            classical_control_value=0b10,
        )
        qc.add_gate(
            X,
            targets=dq[1],
            classical_controls=[0, 1],
            classical_control_value=0b11,
        )
        qc.add_gate(
            X,
            targets=dq[2],
            classical_controls=[0, 1],
            classical_control_value=0b01,
        )

        return qc

    def decode_circuit(self, data_qubits: IntSequence) -> QubitCircuit:
        """
        Constructs the decoding circuit which is the inverse of the encoding operation,
        used to recover the original logical qubit. TOFFOLI gate verifies parity.

        Parameters
        ----------
        data_qubits : sequence of int
            Indices of 3 data qubits.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            The decoding quantum circuit.

        Raises
        ------
        ValueError: If the number of data qubits is not 3.
        """
        if len(data_qubits) != self.n_data:
            raise ValueError(
                f"Expected {self.n_data} data qubits, got {len(data_qubits)}."
            )
        qc = QubitCircuit(max(data_qubits) + 1)
        control = data_qubits[0]
        for target in reversed(data_qubits[1:]):
            qc.add_gate(CX, controls=control, targets=target)

        qc.add_gate(TOFFOLI, controls=data_qubits[1:], targets=control)
        return qc
