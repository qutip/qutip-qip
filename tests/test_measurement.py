import numpy as np
import pytest
from math import sqrt
from qutip_qip.operations.measurement import Mz
from qutip import basis, ket2dm, tensor, rand_ket
import qutip


@pytest.mark.repeat(10)
def test_measurement_comp_basis():
    """
    Test measurements to test probability calculation in
    computational basis measurements on a 3 qubit state
    """

    qubit_kets = [rand_ket(2), rand_ket(2), rand_ket(2)]
    qubit_dms = [ket2dm(qubit_kets[i]) for i in range(3)]

    state = tensor(qubit_kets)
    density_mat = tensor(qubit_dms)

    for i in range(3):
        final_states, probabilities_state = Mz.measurement_comp_basis(state, [i])
        final_dms, probabilities_dm = Mz.measurement_comp_basis(density_mat, [i])

        amps = qubit_kets[i].full()
        probabilities_i = [np.abs(amps[0][0]) ** 2, np.abs(amps[1][0]) ** 2]

        np.testing.assert_allclose(probabilities_state, probabilities_dm)
        np.testing.assert_allclose(probabilities_state, probabilities_i)
        for j, final_state in enumerate(final_states):
            np.testing.assert_allclose(final_dms[j].full(), ket2dm(final_state).full())


@pytest.mark.parametrize("index", [0, 1])
def test_measurement_collapse(index):
    """
    Test if correct state is created after measurement using the example of
    the Bell state
    """

    state_00 = tensor(basis(2, 0), basis(2, 0))
    state_11 = tensor(basis(2, 1), basis(2, 1))

    bell_state = (state_00 + state_11) / sqrt(2)

    states, probabilities = Mz.measurement_comp_basis(bell_state, targets=[index])
    np.testing.assert_allclose(probabilities, [0.5, 0.5])

    for i, state in enumerate(states):
        if i == 0:
            states_00, probability_00 = Mz.measurement_comp_basis(
                state, targets=[1 - index]
            )
            assert probability_00[0] == 1
            assert states_00[1] is None
        else:
            states_11, probability_11 = Mz.measurement_comp_basis(
                state, targets=[1 - index]
            )
            assert probability_11[1] == 1
            assert states_11[0] is None


def test_against_numerical_error():
    state = qutip.Qobj([[1], [1.0e-12]])
    states, probabilities = Mz.measurement_comp_basis(state, [0])
    assert states[1] is None
    assert probabilities[1] == 0.0
