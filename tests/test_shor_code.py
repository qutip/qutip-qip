import pytest
from qutip_qip.algorithms import ShorCode


def test_shor_encode_structure():
    shor = ShorCode()
    circuit = shor.encode_circuit()

    snot_targets = [
        gate.targets[0] for gate in circuit.gates if gate.name == "SNOT"
    ]
    assert set(snot_targets) == {
        0,
        3,
        6,
    }, "Hadamards should be on qubits 0, 3, and 6."

    cnot_gates = [gate for gate in circuit.gates if gate.name == "CNOT"]

    expected_blocks = [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8)]
    actual_blocks = [(g.controls[0], g.targets[0]) for g in cnot_gates]
    for pair in expected_blocks:
        assert pair in actual_blocks, f"CNOT {pair} not found in circuit."
