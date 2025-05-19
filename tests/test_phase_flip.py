import pytest
from qutip_qip.algorithms import PhaseFlipCode  # Replace with actual module name


def test_encode_circuit_structure():
    code = PhaseFlipCode()
    circuit = code.encode_circuit()
    expected = [
        ("SNOT", [], [0]),
        ("SNOT", [], [1]),
        ("SNOT", [], [2]),
        ("CNOT", [0], [1]),
        ("CNOT", [0], [2]),
    ]
    for gate, (name, controls, targets) in zip(circuit.gates, expected):
        assert gate.name == name
        assert gate.targets == targets


def test_syndrome_measurement_circuit_structure():
    code = PhaseFlipCode()
    circuit = code.syndrome_measurement_circuit()

    # 3 Hadamards + 4 CNOTs + 3 Hadamards = 10 gates
    assert len(circuit.gates) == 10
    assert circuit.gates[0].name == "SNOT"
    assert circuit.gates[3].name == "CNOT"
    assert circuit.gates[7].name == "SNOT"


@pytest.mark.parametrize("syndrome,expected_target", [
    ((1, 0), 0),
    ((1, 1), 1),
    ((0, 1), 2),
    ((0, 0), None),
])
def test_correction_circuit_behavior(syndrome, expected_target):
    code = PhaseFlipCode()
    circuit = code.correction_circuit(syndrome)

    if expected_target is None:
        assert len(circuit.gates) == 0
    else:
        assert len(circuit.gates) == 1
        gate = circuit.gates[0]
        assert gate.name == "Z"
        assert gate.targets == [expected_target]


def test_decode_circuit_structure():
    code = PhaseFlipCode()
    circuit = code.decode_circuit()

    assert circuit.gates[-1].name == "SNOT"
    assert circuit.gates[-3].name == "SNOT"
    assert all(gate.name == "SNOT" for gate in circuit.gates[-3:])
