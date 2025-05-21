import pytest
from qutip_qip.circuit import Gate
from qutip_qip.algorithms import (
    BitFlipCode,
)  # Update with the actual module name


def test_encode_circuit_structure():
    code = BitFlipCode(data_qubits=[0, 1, 2], syndrome_qubits=[3, 4])
    circuit = code.encode_circuit()

    assert len(circuit.gates) == 2
    assert circuit.gates[0].name == "CNOT"
    assert circuit.gates[0].controls == [0]
    assert circuit.gates[0].targets == [1]
    assert circuit.gates[1].controls == [0]
    assert circuit.gates[1].targets == [2]


def test_syndrome_circuit_structure():
    code = BitFlipCode(data_qubits=[0, 1, 2], syndrome_qubits=[3, 4])
    circuit = code.syndrome_measurement_circuit()

    expected = [
        ("CNOT", [0], [3]),
        ("CNOT", [1], [3]),
        ("CNOT", [1], [4]),
        ("CNOT", [2], [4]),
    ]

    for gate, (name, controls, targets) in zip(circuit.gates, expected):
        assert gate.name == name
        assert gate.controls == controls
        assert gate.targets == targets


@pytest.mark.parametrize(
    "syndrome,expected_target",
    [
        ((1, 0), 0),
        ((1, 1), 1),
        ((0, 1), 2),
        ((0, 0), None),
    ],
)
def test_correction_circuit_behavior(syndrome, expected_target):
    code = BitFlipCode(data_qubits=[0, 1, 2], syndrome_qubits=[3, 4])
    circuit = code.correction_circuit(syndrome)

    if expected_target is None:
        assert len(circuit.gates) == 0
    else:
        assert len(circuit.gates) == 1
        gate = circuit.gates[0]
        assert gate.name == "X"
        assert gate.targets == [expected_target]


def test_decode_circuit_structure():
    code = BitFlipCode(data_qubits=[0, 1, 2], syndrome_qubits=[3, 4])
    circuit = code.decode_circuit()

    assert len(circuit.gates) == 3
    assert circuit.gates[0].name == "CNOT"
    assert circuit.gates[0].controls == [0]
    assert circuit.gates[0].targets == [2]
    assert circuit.gates[1].controls == [0]
    assert circuit.gates[1].targets == [1]
    assert circuit.gates[2].name == "TOFFOLI"
    assert circuit.gates[2].controls == [1, 2]
    assert circuit.gates[2].targets == [0]
