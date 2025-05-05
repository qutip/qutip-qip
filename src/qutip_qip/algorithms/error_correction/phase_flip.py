import numpy as np
from qutip import Qobj, tensor, basis, sigmax, sigmay, sigmaz
from qutip_qip.operations import Gate
from qutip_qip.circuit import QubitCircuit

__all__ = ["PhaseFlipCode"]


class PhaseFlipCode:
    """
    Implementation of the 3-qubit phase-flip code.
    The phase-flip code protects against Z (phase-flip) errors by encoding
    a single logical qubit across three physical qubits:
    |0⟩ → |+++⟩
    |1⟩ → |---⟩

    This is accomplished by applying Hadamard gates to transform the bit-flip code
    into the phase-flip code, as phase errors in the Hadamard basis appear as bit-flips.
    """

    @staticmethod
    def encode_circuit():
        """
        Create a circuit for encoding a single qubit into the 3-qubit phase-flip code.

        Returns
        -------
        qc : instance of QubitCircuit
            Encoding circuit for the phase-flip code.
        """
        qc = QubitCircuit(3)

        # First apply Hadamard to the input qubit
        qc.add_gate("SNOT", targets=0)

        # Then use the bit-flip encoding structure
        qc.add_gate("CNOT", controls=0, targets=1)
        qc.add_gate("CNOT", controls=0, targets=2)

        # Apply Hadamard gates to all qubits
        qc.add_gate("SNOT", targets=0)
        qc.add_gate("SNOT", targets=1)
        qc.add_gate("SNOT", targets=2)

        return qc

    @staticmethod
    def syndrome_measurement_circuit():
        """
        Create a circuit for syndrome measurement of the 3-qubit phase-flip code.

        Returns
        -------
        qc : instance of QubitCircuit
            Syndrome measurement circuit that uses two ancilla qubits.
        """
        qc = QubitCircuit(5)  # 3 data qubits + 2 syndrome qubits

        qc.add_gate("SNOT", targets=0)
        qc.add_gate("SNOT", targets=1)
        qc.add_gate("SNOT", targets=2)

        qc.add_gate("CNOT", controls=0, targets=3)
        qc.add_gate("CNOT", controls=1, targets=3)

        qc.add_gate("CNOT", controls=1, targets=4)
        qc.add_gate("CNOT", controls=2, targets=4)

        qc.add_gate("SNOT", targets=0)
        qc.add_gate("SNOT", targets=1)
        qc.add_gate("SNOT", targets=2)

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
            Correction circuit applying Z gates as needed.
        """
        qc = QubitCircuit(3)
        s1, s2 = syndrome

        # Syndrome interpretation:
        # s1=0, s2=0: No error
        # s1=1, s2=0: Error on qubit 0
        # s1=1, s2=1: Error on qubit 1
        # s1=0, s2=1: Error on qubit 2

        if s1 == 1 and s2 == 0:
            qc.add_gate("Z", targets=0)
        elif s1 == 1 and s2 == 1:
            qc.add_gate("Z", targets=1)
        elif s1 == 0 and s2 == 1:
            qc.add_gate("Z", targets=2)

        return qc

    @staticmethod
    def decode_circuit():
        """
        Create a circuit for decoding the 3-qubit phase-flip code.

        Returns
        -------
        qc : instance of QubitCircuit
            Decoding circuit for the phase-flip code.
        """
        qc = QubitCircuit(3)

        qc.add_gate("SNOT", targets=0)
        qc.add_gate("SNOT", targets=1)
        qc.add_gate("SNOT", targets=2)

        qc.add_gate("CNOT", controls=0, targets=2)
        qc.add_gate("CNOT", controls=0, targets=1)

        qc.add_gate("TOFFOLI", controls=[1, 2], targets=0)

        qc.add_gate("SNOT", targets=0)

        return qc
