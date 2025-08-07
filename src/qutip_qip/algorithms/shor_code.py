from qutip_qip.circuit import QubitCircuit
from qutip_qip.algorithms import BitFlipCode, PhaseFlipCode

__all__ = ["ShorCode"]


class ShorCode:
    """
    Constructs the 9-qubit Shor code encoding circuit using BitFlipCode and PhaseFlipCode.

    The Shor code protects against arbitrary single-qubit errors by combining
    bit-flip and phase-flip redundancy encoding.
    """

    def __init__(self):
        self.n_qubits = 9  # Total qubits in the Shor code

    def encode_circuit(self):
        """
        Construct the 9-qubit Shor code encoding circuit.

        Returns:
            QubitCircuit: Circuit that encodes one logical qubit into the Shor code.
        """
        qc = QubitCircuit(N=self.n_qubits)

        # Step 1: Bit-flip encode qubit 0 → [0, 1, 2]
        bit_code = BitFlipCode()
        bit_encode = bit_code.encode_circuit([0, 1, 2])
        qc.gates.extend(bit_encode.gates)

        # Step 2: Phase-flip encode each of [0,1,2] across 3 qubits each:
        phase_blocks = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

        for block in phase_blocks:
            phase_code = PhaseFlipCode()
            phase_encode = phase_code.encode_circuit(block)
            qc.gates.extend(phase_encode.gates)

        return qc
