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
        assert(qubit_states([0]) == basis(2, 0))
        assert(qubit_states([1]) == basis(2, 1))
        assert(qubit_states([0, 1]) == tensor(basis(2, 0), basis(2, 1)))
        plus = (basis(2, 0) + basis(2, 1)).unit()
        minus = (basis(2, 0) - basis(2, 1)).unit()
        assert(qubit_states("-+") == tensor(minus, plus))
        assert(qubit_states("0+") == tensor(basis(2, 0), plus))
        assert(qubit_states("+11") == tensor(plus, basis(2, 1), basis(2, 1)))
        assert(
            qubit_states([1.j/np.sqrt(2), 1.]) ==
            tensor(qutip.Qobj([[1/np.sqrt(2)], [1.j/np.sqrt(2)]]), basis(2, 1))
            )

    @pytest.mark.parametrize(
        "state, full_dims",
        [
            (qutip.rand_dm([3, 2, 3]), [3, 2, 3]),
            (qutip.rand_ket([2, 3, 3]), [2, 3, 3]),
            (
                qutip.rand_ket([3, 2, 3]).dag(),
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
            (qutip.rand_dm([2, 2, 2]), [3, 2, 3]),
            (qutip.rand_ket([2, 2, 2]), [2, 3, 3]),
            (
                qutip.rand_ket([2, 2, 2]).dag(),
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
