import pytest
from qutip_qip.algorithms import ShorCode

def test_encode_circuit():
    """Test the Shor code encoding circuit structure."""
    qc = ShorCode.encode_circuit()
    
    # Check correct number of qubits
    assert qc.N == 9
    
    # Check total number of gates (1H + 2CNOT + 3H + 6CNOT = 12 gates)
    assert len(qc.gates) == 12
    
    # Check first Hadamard gate (phase-flip encoding starts)
    assert qc.gates[0].name == "SNOT" and qc.gates[0].targets[0] == 0
    
    # Check first level CNOTs (create GHZ-like state across blocks)
    assert qc.gates[1].name == "CNOT"
    assert qc.gates[1].controls == [0] and qc.gates[1].targets[0] == 3
    assert qc.gates[2].name == "CNOT"
    assert qc.gates[2].controls == [0] and qc.gates[2].targets[0] == 6
    
    # Check three Hadamard gates (complete phase-flip encoding)
    assert qc.gates[3].name == "SNOT" and qc.gates[3].targets[0] == 0
    assert qc.gates[4].name == "SNOT" and qc.gates[4].targets[0] == 3
    assert qc.gates[5].name == "SNOT" and qc.gates[5].targets[0] == 6
    
    # Check bit-flip encoding CNOTs for first block
    assert qc.gates[6].name == "CNOT" 
    assert qc.gates[6].controls == [0] and qc.gates[6].targets[0] == 1
    assert qc.gates[7].name == "CNOT"
    assert qc.gates[7].controls == [0] and qc.gates[7].targets[0] == 2
    
    # Check bit-flip encoding CNOTs for second block
    assert qc.gates[8].name == "CNOT"
    assert qc.gates[8].controls == [3] and qc.gates[8].targets[0] == 4
    assert qc.gates[9].name == "CNOT"
    assert qc.gates[9].controls == [3] and qc.gates[9].targets[0] == 5
    
    # Check bit-flip encoding CNOTs for third block
    assert qc.gates[10].name == "CNOT"
    assert qc.gates[10].controls == [6] and qc.gates[10].targets[0] == 7
    assert qc.gates[11].name == "CNOT"
    assert qc.gates[11].controls == [6] and qc.gates[11].targets[0] == 8
