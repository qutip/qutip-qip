import pytest
import qutip
import numpy as np
from qutip_qip.circuit import QubitCircuit
from qutip_qip.algorithms import PhaseFlipCode


@pytest.fixture
def code():
    return PhaseFlipCode()


@pytest.fixture
def data_qubits():
    return [0, 1, 2]


@pytest.fixture
def syndrome_qubits():
    return [3, 4]


def test_encode_circuit_structure(code, data_qubits):
    qc = code.encode_circuit(data_qubits)
    gate_names = [g.name for g in qc.gates]
    assert gate_names.count("H") == 3
    assert gate_names.count("CNOT") == 2
    assert qc.gates[0].controls == [0]
    assert qc.gates[0].targets == [1]
    assert qc.gates[1].controls == [0]
    assert qc.gates[1].targets == [2]


def test_decode_circuit_structure(code, data_qubits):
    qc = code.decode_circuit(data_qubits)
    gate_names = [g.name for g in qc.gates]
    assert gate_names.count("CNOT") == 2
    assert gate_names.count("H") == 3
    assert qc.gates[3].controls == [0]
    assert qc.gates[3].targets == [2]
    assert qc.gates[4].controls == [0]
    assert qc.gates[4].targets == [1]


@pytest.mark.parametrize("seed", [42, 123, 777, 1337, 9999])
@pytest.mark.parametrize("error_qubit", [None, 0, 1, 2])
def test_phaseflip_correction_simulation(
    code, data_qubits, syndrome_qubits, error_qubit, seed
):
    """
    Simulate the full encoding, Z-error, correction, and decoding process.
    Test across all possible single-qubit errors and multiple random initial states.
    """
    rng = np.random.default_rng(seed)

    # Random initial qubit state
    psi = qutip.rand_ket(2, seed=rng)

    # Full system: qubit + 2 redundant qubits + 2 ancillas
    state = qutip.tensor([psi] + [qutip.basis(2, 0)] * 4)

    # Encode in X-basis
    qc_encode = code.encode_circuit(data_qubits)
    state = qc_encode.run(state)

    # Inject error
    if error_qubit is not None:
        qc_error = QubitCircuit(num_qubits=5)
        qc_error.add_gate("Z", targets=[error_qubit])
        state = qc_error.run(state)

    # Syndrome measurement and Z correction
    qc_correct = code.syndrome_and_correction_circuit(
        data_qubits, syndrome_qubits
    )
    state = qc_correct.run(state)

    # Decode to return to original basis
    qc_decode = code.decode_circuit(data_qubits)
    state = qc_decode.run(state)

    # Extract logical qubit (0th qubit)
    final = state.ptrace(0)
    fidelity = qutip.fidelity(psi, final)

    assert fidelity > 0.99, (
        f"Failed on random seed {seed} on error qubit {error_qubit}. "
        f"Fidelity: {fidelity:.4f}"
    )
