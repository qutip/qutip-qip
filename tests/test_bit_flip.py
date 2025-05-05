import pytest
import numpy as np
from qutip_qip.algorithms import BitFlipCode

def test_encode_circuit_structure():
    """Test that the encoding circuit has the correct structure."""
    qc = BitFlipCode.encode_circuit()
    
    # Check circuit has correct number of qubits
    assert qc.N == 3
    
    # Check it has 2 CNOT gates
    assert len(qc.gates) == 2
    assert all(gate.name == "CNOT" for gate in qc.gates)
    
    # Check gate connections - QubitCircuit uses lists for controls and targets
    assert qc.gates[0].controls == [0]
    assert qc.gates[0].targets[0] == 1
    assert qc.gates[1].controls == [0]
    assert qc.gates[1].targets[0] == 2

def test_syndrome_measurement_circuit_structure():
    """Test that the syndrome measurement circuit has the correct structure."""
    qc = BitFlipCode.syndrome_measurement_circuit()
    
    # Check circuit has correct number of qubits (3 data + 2 syndrome)
    assert qc.N == 5
    
    # Check it has 4 CNOT gates
    assert len(qc.gates) == 4
    assert all(gate.name == "CNOT" for gate in qc.gates)
    
    # Check gate connections for syndrome measurement
    assert qc.gates[0].controls == [0] and qc.gates[0].targets[0] == 3
    assert qc.gates[1].controls == [1] and qc.gates[1].targets[0] == 3
    assert qc.gates[2].controls == [1] and qc.gates[2].targets[0] == 4
    assert qc.gates[3].controls == [2] and qc.gates[3].targets[0] == 4

def test_correction_circuit_no_error():
    """Test correction circuit with no error (syndrome 00)."""
    qc = BitFlipCode.correction_circuit((0, 0))
    assert qc.N == 3
    assert len(qc.gates) == 0  # No correction needed

def test_correction_circuit_qubit0_error():
    """Test correction circuit with error on qubit 0 (syndrome 10)."""
    qc = BitFlipCode.correction_circuit((1, 0))
    assert qc.N == 3
    assert len(qc.gates) == 1
    assert qc.gates[0].name == "X" and qc.gates[0].targets[0] == 0

def test_correction_circuit_qubit1_error():
    """Test correction circuit with error on qubit 1 (syndrome 11)."""
    qc = BitFlipCode.correction_circuit((1, 1))
    assert qc.N == 3
    assert len(qc.gates) == 1
    assert qc.gates[0].name == "X" and qc.gates[0].targets[0] == 1

def test_correction_circuit_qubit2_error():
    """Test correction circuit with error on qubit 2 (syndrome 01)."""
    qc = BitFlipCode.correction_circuit((0, 1))
    assert qc.N == 3
    assert len(qc.gates) == 1
    assert qc.gates[0].name == "X" and qc.gates[0].targets[0] == 2

def test_decode_circuit_structure():
    """Test that the decoding circuit has the correct structure."""
    qc = BitFlipCode.decode_circuit()
    
    # Check circuit has correct number of qubits
    assert qc.N == 3
    
    # Check it has 2 CNOT gates and 1 TOFFOLI gate
    assert len(qc.gates) == 3
    assert qc.gates[0].name == "CNOT"
    assert qc.gates[1].name == "CNOT"
    assert qc.gates[2].name == "TOFFOLI"
    
    # Check gate connections
    assert qc.gates[0].controls == [0] and qc.gates[0].targets[0] == 2
    assert qc.gates[1].controls == [0] and qc.gates[1].targets[0] == 1
    assert qc.gates[2].controls == [1, 2] and qc.gates[2].targets[0] == 0
