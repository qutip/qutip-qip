# bit_flip_code.py

import numpy as np
from qutip import Qobj, tensor, basis, sigmax, sigmay
from qutip_qip.operations import Gate
from qutip_qip.circuit import QubitCircuit

__all__ = ["BitFlipCode"]


class BitFlipCode:
    """
    Implementation of the 3-qubit bit-flip code.

    The bit-flip code protects against X (bit-flip) errors by encoding
    a single logical qubit across three physical qubits:
    |0⟩ → |000⟩
    |1⟩ → |111⟩
    """

    @staticmethod
    def encode_circuit():
        """
        Create a circuit for encoding a single qubit into the 3-qubit bit-flip code.

        Returns
        -------
        qc : instance of QubitCircuit
            Encoding circuit for the bit-flip code.
        """
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("CNOT", controls=0, targets=2)
        return qc

    @staticmethod
    def syndrome_measurement_circuit():
        """
        Create a circuit for syndrome measurement of the 3-qubit bit-flip code.

        Returns
        -------
        qc : instance of QubitCircuit
            Syndrome measurement circuit that uses two ancilla qubits.
        """
        qc = QubitCircuit(5)  # 3 data qubits + 2 syndrome qubits

        # First syndrome measurement: Parity of qubits 0 and 1
        qc.add_gate("CNOT", controls=0, targets=3)
        qc.add_gate("CNOT", controls=1, targets=3)

        # Second syndrome measurement: Parity of qubits 1 and 2
        qc.add_gate("CNOT", controls=1, targets=4)
        qc.add_gate("CNOT", controls=2, targets=4)

        return qc

    @staticmethod
    def correction_circuit(syndrome):
        """
        Create a circuit for error correction based on syndrome measurement.

        Parameters
        ----------
        syndrome : tuple
            Two-bit syndrome measurement result (s1, s2).

        Returns
        -------
        qc : instance of QubitCircuit
            Correction circuit applying X gates as needed.
        """
        qc = QubitCircuit(3)
        s1, s2 = syndrome

        # Syndrome interpretation:
        # s1=0, s2=0: No error
        # s1=1, s2=0: Error on qubit 0
        # s1=1, s2=1: Error on qubit 1
        # s1=0, s2=1: Error on qubit 2

        if s1 == 1 and s2 == 0:
            # Error on qubit 0
            qc.add_gate("X", targets=0)
        elif s1 == 1 and s2 == 1:
            # Error on qubit 1
            qc.add_gate("X", targets=1)
        elif s1 == 0 and s2 == 1:
            # Error on qubit 2
            qc.add_gate("X", targets=2)

        return qc

    @staticmethod
    def decode_circuit():
        """
        Create a circuit for decoding the 3-qubit bit-flip code.

        Returns
        -------
        qc : instance of QubitCircuit
            Decoding circuit for the bit-flip code.
        """
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", controls=0, targets=2)
        qc.add_gate("CNOT", controls=0, targets=1)

        # Add a Toffoli gate to verify the parity
        # If all qubits have the same value, the result is stored in qubit 0
        qc.add_gate("TOFFOLI", controls=[1, 2], targets=0)

        return qc
