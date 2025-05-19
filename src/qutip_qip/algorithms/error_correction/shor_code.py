from qutip_qip.circuit import QubitCircuit
from qutip_qip.algorithms import BitFlipCode
from qutip_qip.algorithms import PhaseFlipCode

class ShorCode:
    """
    Constructs the 9-qubit Shor code encoding circuit using BitFlipCode and PhaseFlipCode.
    """

    def __init__(self):
        # Logical qubit -> Phase protection: [0, 3, 6] (1 qubit to 3)
        # Each → BitFlipCode: [0,1,2], [3,4,5], [6,7,8]
        self.data_qubits = list(range(9))
        self.phase_code = PhaseFlipCode(data_qubits=[0, 3, 6], syndrome_qubits=[])  # No ancillas for encoding
        self.bit_blocks = [
            BitFlipCode(data_qubits=[0, 1, 2], syndrome_qubits=[]),
            BitFlipCode(data_qubits=[3, 4, 5], syndrome_qubits=[]),
            BitFlipCode(data_qubits=[6, 7, 8], syndrome_qubits=[])
        ]
        self.n_qubits = 9  # Encoding only requires 9 qubits

    def encode_circuit(self):
        """
        Construct the 9-qubit Shor code encoding circuit.

        Returns
        -------
        QubitCircuit
            Circuit that encodes one logical qubit into the Shor code.
        """
        qc = QubitCircuit(self.n_qubits)

        phase_encode = self.phase_code.encode_circuit()
        qc.gates.extend(phase_encode.gates)

        for bit_code in self.bit_blocks:
            bit_encode = bit_code.encode_circuit()
            qc.gates.extend(bit_encode.gates)

        return qc
