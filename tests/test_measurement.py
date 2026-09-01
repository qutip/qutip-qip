import numpy as np
import qutip
import pytest
from math import sqrt
from qutip_qip.operations import expand_operator
from qutip_qip.operations.measurement import Measurement, Mz, Mx, My
from qutip import basis, tensor
from qutip.measurement import measurement_statistics


def _get_measurement_results(state, measurement_obj, targets):
    """Helper to apply measurement and return states and probabilities"""
    if isinstance(measurement_obj, type):
        measurement_obj = measurement_obj()
    n = int(np.log2(state.shape[0]))
    raw_ops = measurement_obj.get_measurement_ops()
    expanded_ops = [
        expand_operator(oper=op, dims=[2] * n, targets=targets) for op in raw_ops
    ]

    measurement_tol = qutip.settings.core["atol"] ** 2
    states, probabilities = measurement_statistics(state, expanded_ops)
    probabilities = [p if p > measurement_tol else 0.0 for p in probabilities]
    states = [s if p > measurement_tol else None for s, p in zip(states, probabilities)]
    return states, probabilities


def test_measurement_classes():
    """Test standard measurement class attributes and operators."""
    for cls, name, num_ops in [(Mz, "Mz", 2), (Mx, "Mx", 2), (My, "My", 2)]:
        meas = cls()
        assert meas.name == name
        assert meas.num_qubits == 1
        ops = meas.get_measurement_ops()
        assert len(ops) == num_ops
        for op in ops:
            assert isinstance(op, qutip.Qobj)


def test_custom_measurement_subclass():
    """Test creating a custom Measurement subclass."""

    class MyCustomMeasurement(Measurement):
        def get_measurement_ops(self):
            return [basis(2, 0) * basis(2, 0).dag(), basis(2, 1) * basis(2, 1).dag()]

    meas = MyCustomMeasurement()
    assert meas.name == "M"
    assert meas.num_qubits == 1
    assert len(meas.get_measurement_ops()) == 2


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("measurement_class", [Mz, Mx, My])
def test_measurement_collapse(index, measurement_class):
    """
    Test if correct state is created after measurement using the example of
    the Bell state in respective bases.
    """
    if measurement_class == Mz:
        v0 = basis(2, 0)
        v1 = basis(2, 1)
    elif measurement_class == Mx:
        v0 = (basis(2, 0) + basis(2, 1)).unit()
        v1 = (basis(2, 0) - basis(2, 1)).unit()
    elif measurement_class == My:
        v0 = (basis(2, 0) + 1j * basis(2, 1)).unit()
        v1 = (basis(2, 0) - 1j * basis(2, 1)).unit()

    state_00 = tensor(v0, v0)
    state_11 = tensor(v1, v1)

    bell_state = (state_00 + state_11) / sqrt(2)

    states, probabilities = _get_measurement_results(
        bell_state, measurement_class, targets=[index]
    )
    np.testing.assert_allclose(probabilities, [0.5, 0.5])

    for i, state in enumerate(states):
        if i == 0:
            states_00, probability_00 = _get_measurement_results(
                state, measurement_class, targets=[1 - index]
            )
            assert probability_00[0] == pytest.approx(1.0)
            assert states_00[1] is None
        else:
            states_11, probability_11 = _get_measurement_results(
                state, measurement_class, targets=[1 - index]
            )
            assert probability_11[1] == pytest.approx(1.0)
            assert states_11[0] is None


def test_against_numerical_error():
    state = qutip.Qobj([[1], [1.0e-12]])
    states, probabilities = _get_measurement_results(state, Mz, [0])
    assert states[1] is None
    assert probabilities[1] == 0.0
