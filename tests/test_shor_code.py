import pytest
import qutip
from qutip_qip.algorithms import ShorCode


@pytest.fixture
def code():
    return ShorCode()


def test_shor_circuit_structure(code):
    qc = code.encode_circuit()
    assert qc.N == 9
    assert len(qc.gates) > 0


def test_shor_encodes_zero(code):
    qc = code.encode_circuit()
    zero_state = qutip.tensor([qutip.basis(2, 0)] * 9)
    encoded = qc.run(zero_state)
    assert abs(encoded.norm() - 1.0) < 1e-10


def test_shor_encodes_one(code):
    qc = code.encode_circuit()
    one_input = qutip.tensor([qutip.basis(2, 1)] + [qutip.basis(2, 0)] * 8)
    encoded = qc.run(one_input)
    assert abs(encoded.norm() - 1.0) < 1e-10


def test_shor_linearity(code):
    qc = code.encode_circuit()

    # Encode |0⟩ and |1⟩
    zero_input = qutip.tensor([qutip.basis(2, 0)] + [qutip.basis(2, 0)] * 8)
    one_input = qutip.tensor([qutip.basis(2, 1)] + [qutip.basis(2, 0)] * 8)
    encoded_zero = qc.run(zero_input)
    encoded_one = qc.run(one_input)

    # Encode superposition
    alpha, beta = 0.6, 0.8
    superpos = alpha * qutip.basis(2, 0) + beta * qutip.basis(2, 1)
    superpos_input = qutip.tensor([superpos] + [qutip.basis(2, 0)] * 8)
    encoded_superpos = qc.run(superpos_input)

    # Check linearity
    expected = alpha * encoded_zero + beta * encoded_one
    fidelity = qutip.fidelity(encoded_superpos, expected)
    assert fidelity > 0.99


def test_shor_orthogonality(code):
    qc = code.encode_circuit()

    zero_input = qutip.tensor([qutip.basis(2, 0)] + [qutip.basis(2, 0)] * 8)
    one_input = qutip.tensor([qutip.basis(2, 1)] + [qutip.basis(2, 0)] * 8)

    encoded_zero = qc.run(zero_input)
    encoded_one = qc.run(one_input)

    # Should be orthogonal
    overlap = abs(encoded_zero.overlap(encoded_one))
    assert overlap < 0.1
