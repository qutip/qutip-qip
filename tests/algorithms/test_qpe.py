from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qutip_qip.algorithms import QFT, QPE
from qutip_qip.operations import Inverse, Mz
from qutip_qip.operations.gates import PHASE, S, T, X, Z

assert_close = partial(assert_allclose, atol=1e-12)


def reading(k, num_counting, target_state):
    return np.kron(np.eye(2**num_counting)[k], target_state)


@pytest.mark.parametrize(
    "unitary, num_counting",
    [(Z, 1), (S, 2), (T, 3), (PHASE(2 * np.pi * 3 / 8), 3), (QFT(2), 2)],
    ids=repr,
)
def test_reads_the_phase_of_every_eigenstate(unitary, num_counting, gate_matrix):
    qpe = gate_matrix(QPE(unitary, num_counting))
    values, vectors = np.linalg.eig(unitary.get_qobj().full())
    phases = np.angle(values) / (2 * np.pi)

    for phase, vector in zip(phases, vectors.T):
        k = round(phase * 2**num_counting) % 2**num_counting
        expected = reading(k, num_counting, vector)
        assert_close(qpe @ reading(0, num_counting, vector), expected)


@pytest.mark.parametrize("phase", [0.3, 0.57, 0.95])
def test_inexact_phase_peaks_at_the_nearest_value(phase, gate_matrix):
    qpe = gate_matrix(QPE(PHASE(2 * np.pi * phase), 3))
    out = qpe @ reading(0, 3, [0, 1])
    probability = (abs(out) ** 2).reshape(8, 2).sum(axis=1)
    nearest = round(phase * 8) % 8

    assert probability.argmax() == nearest
    assert probability[nearest] > 4 / np.pi**2


def test_inverse_undoes_qpe(gate_matrix):
    qpe = QPE(PHASE(0.3), 3)
    assert_close(gate_matrix(qpe, Inverse(qpe)), np.eye(2**qpe.num_qreg))


def test_acts_on_counting_and_target_qubits():
    assert QPE(QFT(2), 3).num_qreg == 5


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        QPE(X, 0)
    with pytest.raises(TypeError):
        QPE(Mz, 2)
