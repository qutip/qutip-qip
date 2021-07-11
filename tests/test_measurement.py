import numpy as np
import scipy
import pytest
from math import sqrt
from qutip_qip.circuit import Measurement
from qutip import (Qobj, basis, isequal, ket2dm,
                    sigmax, sigmay, sigmaz, identity, tensor, rand_ket)
from qutip.measurement import (measure_povm, measurement_statistics_povm,
                                measure_observable,
                                measurement_statistics_observable)


@pytest.mark.repeat(10)
def test_measurement_comp_basis():
    """
    Test measurements to test probability calculation in
    computational basis measurments on a 3 qubit state
    """

    qubit_kets = [rand_ket(2), rand_ket(2), rand_ket(2)]
    qubit_dms = [ket2dm(qubit_kets[i]) for i in range(3)]

    state = tensor(qubit_kets)
    density_mat = tensor(qubit_dms)

    for i in range(3):
        m_i = Measurement("M" + str(i), i)
        final_states, probabilities_state = m_i.measurement_comp_basis(state)
        final_dms, probabilities_dm = m_i.measurement_comp_basis(density_mat)

        amps = qubit_kets[i].full()
        probabilities_i = [np.abs(amps[0][0])**2, np.abs(amps[1][0])**2]

        np.testing.assert_allclose(probabilities_state, probabilities_dm)
        np.testing.assert_allclose(probabilities_state, probabilities_i)
        for j, final_state in enumerate(final_states):
            np.testing.assert_allclose(final_dms[j], ket2dm(final_state))


@pytest.mark.parametrize("index", [0, 1])
def test_measurement_collapse(index):
    """
    Test if correct state is created after measurement using the example of
    the Bell state
    """

    state_00 = tensor(basis(2, 0), basis(2, 0))
    state_11 = tensor(basis(2, 1), basis(2, 1))

    bell_state = (state_00 + state_11)/sqrt(2)
    M = Measurement("BM", targets=[index])

    states, probabilities = M.measurement_comp_basis(bell_state)
    np.testing.assert_allclose(probabilities, [0.5, 0.5])

    for i, state in enumerate(states):
        if i == 0:
            Mprime = Measurement("00", targets=[1-index])
            states_00, probability_00 = Mprime.measurement_comp_basis(state)
            assert probability_00[0] == 1
            assert states_00[1] is None
        else:
            Mprime = Measurement("11", targets=[1-index])
            states_11, probability_11 = Mprime.measurement_comp_basis(state)
            assert probability_11[1] == 1
            assert states_11[0] is None
