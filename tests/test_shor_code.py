import pytest
from qutip_qip.algorithms import ShorCode  # Adjust the import as per your file/module name

def test_shor_encode_structure():
    shor = ShorCode()
    circuit = shor.encode_circuit()

    # Gate count: 3 Hadamards (phase flip) + 6 CNOTs (bit flip blocks)
    assert len(circuit.gates) == 9, "Shor code should have 3 Hadamards and 6 CNOTs in encode circuit."

    # Check Hadamard gates applied to qubits 0, 3, 6
    snot_targets = [gate.targets[0] for gate in circuit.gates if gate.name == "SNOT"]
    assert set(snot_targets) == {0, 3, 6}, "Hadamards should be on qubits 0, 3, and 6."

    # Check correct number of CNOTs for 3 blocks
    cnot_gates = [gate for gate in circuit.gates if gate.name == "CNOT"]
    assert len(cnot_gates) == 6, "Shor code encoding must include 6 CNOTs."

    # Check CNOTs in each block
    expected_blocks = [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8)]
    actual_blocks = [(g.controls[0], g.targets[0]) for g in cnot_gates]
    for pair in expected_blocks:
        assert pair in actual_blocks, f"CNOT {pair} not found in circuit."
