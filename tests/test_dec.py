import numpy as np
from qutip_qip.operations import H, RX, RY
from qutip_qip.operations.decomposer import resolve_decomposition


def test_hadamard_decomposition():
    gate = H(0)
    decomposed = resolve_decomposition(gate, basis_1q=["RX", "RY"], basis_2q=["CNOT"])

    assert len(decomposed) == 2
    assert isinstance(decomposed[0], RY)
    assert isinstance(decomposed[1], RX)
    np.testing.assert_allclose(decomposed[0].arg_value, np.pi / 2)
    np.testing.assert_allclose(decomposed[1].arg_value, np.pi)
