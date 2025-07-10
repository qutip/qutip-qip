import pytest
import qutip
from qutip_qip.algorithms import BitFlipCode
from qutip_qip.circuit import QubitCircuit


@pytest.fixture
def code():
    return BitFlipCode()


@pytest.fixture
def data_qubits():
    return [0, 1, 2]


@pytest.fixture
def syndrome_qubits():
    return [3, 4]


def test_encode_circuit_structure(code, data_qubits):
    qc = code.encode_circuit(data_qubits)
    assert len(qc.gates) == 2
    assert qc.gates[0].name == "CNOT"
    assert qc.gates[0].controls == [0]
    assert qc.gates[0].targets == [1]
    assert qc.gates[1].controls == [0]
    assert qc.gates[1].targets == [2]


def test_bitflip_correction(code, data_qubits, syndrome_qubits):
    # Initial random state |ψ⟩ on qubit 0
    psi = qutip.rand_ket(2)

    # Full state: |ψ⟩ ⊗ |0000⟩ (qubits 1,2,3,4)
    state = qutip.tensor([psi] + [qutip.basis(2, 0)] * 4)

    # Step 1: Encode |ψ⟩ over qubits 0,1,2
    qc_encode = code.encode_circuit(data_qubits)
    state = qc_encode.run(state)

    # Step 2: Apply bit-flip error to qubit 0
    qc_error = QubitCircuit(N=5)
    qc_error.add_gate("X", targets=[0])
    state = qc_error.run(state)

    # Step 3: Syndrome + correction
    qc_correct = code.syndrome_and_correction_circuit(
        data_qubits, syndrome_qubits
    )
    state = qc_correct.run(state)

    # Step 4: Decode
    qc_decode = code.decode_circuit(data_qubits)
    state = qc_decode.run(state)

    # Step 5: Trace out ancillas/qubits 1-4, keep logical qubit 0
    final_qubit = state.ptrace(0)

    # Fidelity between original |ψ⟩ and final decoded state
    fidelity = qutip.fidelity(final_qubit, psi)
    assert fidelity > 0.99, f"Fidelity too low: {fidelity:.4f}"
