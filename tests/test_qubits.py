from numpy.testing import assert_, run_module_suite
import numpy as np
import pytest
import qutip
from qutip import tensor, basis
from qutip_qip.qubits import (
    qubit_states,
    truncate_to_qubit_state,
    expand_qubit_state,
)


class TestQubits:
    """
    A test class for the QuTiP functions for qubits.
    """

    def testQubitStates(self):
        """
        Tests the qubit_states function.
        """
        psi0_a = basis(2, 0)
        psi0_b = qubit_states()
        assert_(psi0_a == psi0_b)

        psi1_a = basis(2, 1)
        psi1_b = qubit_states(states=[1])
        assert_(psi1_a == psi1_b)

        psi01_a = tensor(psi0_a, psi1_a)
        psi01_b = qubit_states(N=2, states=[0, 1])
        assert_(psi01_a == psi01_b)

    @pytest.mark.parametrize(
        "state, full_dims",
        [
            (qutip.rand_dm(18, dims=[[3, 2, 3], [3, 2, 3]]), [3, 2, 3]),
            (qutip.rand_ket(18, dims=[[2, 3, 3], [1, 1, 1]]), [2, 3, 3]),
            (
                qutip.Qobj(
                    qutip.rand_ket(18).full().transpose(),
                    dims=[[1, 1, 1], [3, 2, 3]],
                ),
                [3, 2, 3],
            ),
        ],
    )
    def test_state_truncation(self, state, full_dims):
        reduced_state = truncate_to_qubit_state(state)
        for _ in range(5):
            ind1 = np.random.choice([0, 1], len(full_dims))
            ind2 = np.random.choice([0, 1], len(full_dims))
            ind_red1 = np.ravel_multi_index(ind1, [2] * len(full_dims))
            ind_red2 = np.ravel_multi_index(ind2, [2] * len(full_dims))
            ind_full1 = np.ravel_multi_index(ind1, full_dims)
            ind_full2 = np.ravel_multi_index(ind2, full_dims)

            if state.isoper:
                assert (
                    reduced_state[ind_red1, ind_red2]
                    == state[ind_full1, ind_full2]
                )
            elif state.isket:
                assert reduced_state[ind_red1, 0] == state[ind_full1, 0]
            elif state.isbra:
                assert reduced_state[0, ind_red2] == state[0, ind_full2]

    @pytest.mark.parametrize(
        "state, full_dims",
        [
            (qutip.rand_dm(8, dims=[[2, 2, 2], [2, 2, 2]]), [3, 2, 3]),
            (qutip.rand_ket(8, dims=[[2, 2, 2], [1, 1, 1]]), [2, 3, 3]),
            (
                qutip.Qobj(
                    qutip.rand_ket(8).full().transpose(),
                    dims=[[1, 1, 1], [2, 2, 2]],
                ),
                [3, 2, 3],
            ),
        ],
    )
    def test_state_expansion(self, state, full_dims):
        full_state = expand_qubit_state(state, full_dims)
        for _ in range(5):
            ind_red1, ind_red2 = np.random.randint(2 ** len(full_dims), size=2)
            reduced_dims = [2] * len(full_dims)
            ind_full1 = np.ravel_multi_index(
                np.unravel_index(ind_red1, reduced_dims), full_dims
            )
            ind_full2 = np.ravel_multi_index(
                np.unravel_index(ind_red2, reduced_dims), full_dims
            )

            if state.isoper:
                assert (
                    state[ind_red1, ind_red2]
                    == full_state[ind_full1, ind_full2]
                )
            elif state.isket:
                assert state[ind_red1, 0] == full_state[ind_full1, 0]
            elif state.isbra:
                assert state[0, ind_red2] == full_state[0, ind_full2]
