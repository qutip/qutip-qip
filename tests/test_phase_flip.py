import pytest
import numpy as np
from qutip import Qobj, tensor, basis, fidelity
from qutip_qip.algorithms import PhaseFlipCode


def test_encode_circuit_structure():
    """Test that the encoding circuit has the correct structure."""
    qc = PhaseFlipCode.encode_circuit()

    # Check circuit has correct number of qubits
    assert qc.N == 3

    # Check it has correct number of gates (1 H + 2 CNOT + 3 H = 6 gates)
    assert len(qc.gates) == 6

    # Check first gate is Hadamard on qubit 0
    assert qc.gates[0].name == "SNOT"
    assert qc.gates[0].targets[0] == 0

    # Check CNOT gates
    assert qc.gates[1].name == "CNOT"
    assert qc.gates[1].controls == [0]
    assert qc.gates[1].targets[0] == 1

    assert qc.gates[2].name == "CNOT"
    assert qc.gates[2].controls == [0]
    assert qc.gates[2].targets[0] == 2

    # Check final Hadamard gates on all qubits
    assert qc.gates[3].name == "SNOT"
    assert qc.gates[3].targets[0] == 0

    assert qc.gates[4].name == "SNOT"
    assert qc.gates[4].targets[0] == 1

    assert qc.gates[5].name == "SNOT"
    assert qc.gates[5].targets[0] == 2


def test_syndrome_measurement_circuit_structure():
    """Test that the syndrome measurement circuit has the correct structure."""
    qc = PhaseFlipCode.syndrome_measurement_circuit()

    # Check circuit has correct number of qubits (3 data + 2 syndrome)
    assert qc.N == 5

    # Check it has the correct number of gates (3 H + 4 CNOT + 3 H = 10 gates)
    assert len(qc.gates) == 10

    # Check first three Hadamard gates on data qubits
    for i in range(3):
        assert qc.gates[i].name == "SNOT"
        assert qc.gates[i].targets[0] == i

    # Check CNOT gates for syndrome measurement
    assert qc.gates[3].name == "CNOT"
    assert qc.gates[3].controls == [0] and qc.gates[3].targets[0] == 3

    assert qc.gates[4].name == "CNOT"
    assert qc.gates[4].controls == [1] and qc.gates[4].targets[0] == 3

    assert qc.gates[5].name == "CNOT"
    assert qc.gates[5].controls == [1] and qc.gates[5].targets[0] == 4

    assert qc.gates[6].name == "CNOT"
    assert qc.gates[6].controls == [2] and qc.gates[6].targets[0] == 4

    # Check final three Hadamard gates on data qubits
    for i in range(3):
        assert qc.gates[7 + i].name == "SNOT"
        assert qc.gates[7 + i].targets[0] == i


def test_correction_circuit_no_error():
    """Test correction circuit with no error (syndrome 00)."""
    qc = PhaseFlipCode.correction_circuit((0, 0))
    assert qc.N == 3
    assert len(qc.gates) == 0  # No correction needed


def test_correction_circuit_qubit0_error():
    """Test correction circuit with error on qubit 0 (syndrome 10)."""
    qc = PhaseFlipCode.correction_circuit((1, 0))
    assert qc.N == 3
    assert len(qc.gates) == 1
    assert qc.gates[0].name == "Z" and qc.gates[0].targets[0] == 0


def test_correction_circuit_qubit1_error():
    """Test correction circuit with error on qubit 1 (syndrome 11)."""
    qc = PhaseFlipCode.correction_circuit((1, 1))
    assert qc.N == 3
    assert len(qc.gates) == 1
    assert qc.gates[0].name == "Z" and qc.gates[0].targets[0] == 1


def test_correction_circuit_qubit2_error():
    """Test correction circuit with error on qubit 2 (syndrome 01)."""
    qc = PhaseFlipCode.correction_circuit((0, 1))
    assert qc.N == 3
    assert len(qc.gates) == 1
    assert qc.gates[0].name == "Z" and qc.gates[0].targets[0] == 2


def test_decode_circuit_structure():
    """Test that the decoding circuit has the correct structure."""
    qc = PhaseFlipCode.decode_circuit()

    # Check circuit has correct number of qubits
    assert qc.N == 3

    # Check it has the correct number of gates (3 H + 2 CNOT + 1 TOFFOLI + 1 H = 7 gates)
    assert len(qc.gates) == 7

    # Check first three Hadamard gates
    for i in range(3):
        assert qc.gates[i].name == "SNOT"
        assert qc.gates[i].targets[0] == i

    # Check the two CNOT gates
    assert qc.gates[3].name == "CNOT"
    assert qc.gates[3].controls == [0] and qc.gates[3].targets[0] == 2

    assert qc.gates[4].name == "CNOT"
    assert qc.gates[4].controls == [0] and qc.gates[4].targets[0] == 1

    # Check the TOFFOLI gate
    assert qc.gates[5].name == "TOFFOLI"
    assert qc.gates[5].controls == [1, 2] and qc.gates[5].targets[0] == 0

    # Check the final Hadamard gate
    assert qc.gates[6].name == "SNOT"
    assert qc.gates[6].targets[0] == 0
