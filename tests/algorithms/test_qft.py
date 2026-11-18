from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import block_diag

from qutip_qip.algorithms import QFT
from qutip_qip.operations import Controlled, Inverse

assert_close = partial(assert_allclose, atol=1e-12)


@pytest.fixture(params=[1, 2, 3, 4])
def qft(request):
    return QFT(request.param)


def matrix(op):
    return op.get_qobj().full()


def test_matrix_is_the_fourier_transform(qft):
    # QFT|x> = sum_k exp(2 pi i x k / N) |k> / sqrt(N). numpy calls this sign
    # of the exponent the inverse FFT.
    assert_close(matrix(qft), np.fft.ifft(np.eye(2**qft.n), norm="ortho"))


def test_gates_build_the_matrix(qft, gate_matrix):
    assert_close(gate_matrix(qft), matrix(qft))


def test_inverse_undoes_qft(qft, gate_matrix):
    identity = np.eye(2**qft.n)

    assert_close(matrix(Inverse(qft)) @ matrix(qft), identity)
    assert_close(gate_matrix(qft, Inverse(qft)), identity)
    assert_close(gate_matrix(Inverse(qft), qft), identity)


def test_controlled_applies_qft_only_when_the_control_is_one(qft, gate_matrix):
    expected = block_diag(np.eye(2**qft.n), matrix(qft))
    assert_close(gate_matrix(Controlled(qft)), expected)


def test_shift_becomes_a_phase(qft, gate_matrix):
    # Adding 1 (mod N) before the transform is the same as multiplying |k> by
    # exp(2 pi i k / N) after it. Phase estimation relies on this.
    N = 2**qft.n
    add_one = np.roll(np.eye(N), 1, axis=0)
    phase = np.diag(np.exp(2j * np.pi * np.arange(N) / N))

    assert_close(gate_matrix(qft) @ add_one, phase @ gate_matrix(qft))


def test_wrappers_of_the_family():
    # Inverse(QFT) and Controlled(QFT) are families too, called like QFT.
    assert Inverse(QFT)(3) == Inverse(QFT(3))
    assert Controlled(QFT)(3) == Controlled(QFT(3))
    assert Inverse(Inverse(QFT)) is QFT
    assert Inverse(Inverse(QFT(3))) == QFT(3)


def test_needs_at_least_one_qubit():
    with pytest.raises(ValueError):
        QFT(0)
