from numpy.testing import assert_equal, assert_allclose
from qutip import basis, tensor
from qutip_qip.algorithms.deutsch_jozsa import deutsch_jozsa, dj_oracle


class TestDeutschJosza:
    def test_circuit_size(self):
        """Verify the circuit handles the ancilla qubit correctly."""
        n_qubits = 4
        oracle = dj_oracle(n_qubits, case="constant")
        qc = deutsch_jozsa(n_qubits, oracle)
        assert_equal(qc.num_qubits, 5)

    def test_constant_oracle_result(self):
        """
        Test that a constant oracle results in measuring |00..0>
        with 100% probability.
        """
        n_qubits = 3
        oracle = dj_oracle(3, "constant")
        qc = deutsch_jozsa(3, oracle)

        initial_state = tensor([basis(2, 0)] * (n_qubits + 1))

        U_dj = qc.compute_unitary()
        final_state = U_dj * initial_state

        prob_0000 = (
            abs(final_state.overlap(tensor([basis(2, 0)] * (n_qubits + 1))))
            ** 2
        )
        prob_0001 = (
            abs(
                final_state.overlap(
                    tensor([basis(2, 0)] * n_qubits + [basis(2, 1)])
                )
            )
            ** 2
        )

        assert_allclose(prob_0000 + prob_0001, 1.0, atol=1e-7)

    def test_balanced_oracle_result(self):
        """
        Test that a balanced oracle results in 0% probability of measuring |00...0>.
        """
        n_qubits = 2
        oracle = dj_oracle(n_qubits, case="balanced")
        qc = deutsch_jozsa(n_qubits, oracle)

        initial_state = tensor([basis(2, 0)] * (n_qubits + 1))
        U_dj = qc.compute_unitary()
        final_state = U_dj * initial_state

        prob_00_reg = (
            abs(final_state.overlap(tensor([basis(2, 0)] * (n_qubits + 1))))
            ** 2
            + abs(
                final_state.overlap(
                    tensor([basis(2, 0)] * n_qubits + [basis(2, 1)])
                )
            )
            ** 2
        )

        assert_allclose(prob_00_reg, 0.0, atol=1e-7)
