import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import qutip
from qutip_qip.device import (
    Model,
    Processor,
    DispersiveCavityQED,
    LinearSpinChain,
    CircularSpinChain,
    SCQubits,
)
from qutip_qip.noise import RelaxationNoise


def test_cavityqed_model():
    model = DispersiveCavityQED(3, epsmax=[1.1, 1, 0, 0.8], w0=7.0)
    assert model.get_all_drift() == []
    model.get_control_labels()
    model.get_control("g1")
    model.get_control("sx0")
    assert_array_equal(model.params["deltamax"], np.array([1.0] * 3))
    assert_array_equal(model.params["w0"], 7.0)
    assert_array_equal(model.params["epsmax"], [1.1, 1, 0, 0.8])
    assert model.get_control(0) == model.get_control("sx0")
    model.get_control_latex()


@pytest.mark.parametrize(("model_class"), [LinearSpinChain, CircularSpinChain])
def test_spinchain_model(model_class):
    model = LinearSpinChain(3, sx=[1.1, 1, 0, 0.8], sz=7.0, t1=10.0)
    assert model.get_all_drift() == []
    model.get_control_labels()
    if isinstance(model, LinearSpinChain):
        assert len(model.get_control_labels()) == 3 * 3 - 1
    elif isinstance(model, CircularSpinChain):
        assert len(model.get_control_labels()) == 3 * 3
    model.get_control("g1")
    model.get_control("sx0")
    assert_array_equal(model.params["sz"], 7.0)
    assert_array_equal(model.params["sx"], [1.1, 1, 0, 0.8])
    assert model.get_control(0) == model.get_control("sx0")
    model.get_control_latex()
    assert model.params["t1"] == 10.0


def test_scqubits_model():
    model = SCQubits(
        3, dims=[4, 3, 4], omega_single=0.02, alpha=[-0.02, -0.015, -0.025]
    )
    assert model.dims == [4, 3, 4]
    model.get_all_drift()
    model.get_control_labels()
    model.get_control("sx2")
    model.get_control("zx10")
    assert_array_equal(model.params["omega_single"], np.array([0.02] * 3))
    assert_array_equal(model.params["alpha"], [-0.02, -0.015, -0.025])
    assert model.get_control(0) == model.get_control("sx0")
    model.get_control_latex()


def test_define_model_in_processor():
    processor = Processor(3)

    # Define Hamiltonian model
    processor.add_drift(qutip.sigmax(), 1)
    assert processor.get_all_drift() == [(qutip.sigmax(), [1])]
    processor.add_drift(qutip.sigmaz(), cyclic_permutation=True)
    assert len(processor.drift) == 4
    processor.add_control(qutip.sigmaz(), targets=2, label="sz")
    processor.set_coeffs({"sz": np.array([[0.0, 1.0]])})
    processor.controls[0] == qutip.tensor(
        [qutip.qeye(2), qutip.qeye(2), qutip.sigmax()]
    )
    processor.add_control(qutip.sigmax(), label="sx", cyclic_permutation=True)
    label_dict = {"sz", ("sx", (0,)), ("sx", (1,)), ("sx", (2,))}
    for label in processor.get_control_labels():
        assert label in label_dict
    assert processor.get_control("sz") == (qutip.sigmaz(), [2])

    # Relaxation time
    processor.t1 = 100.0
    processor.t2 = 60.0
    assert processor.t1 == 100.0
    assert processor.t2 == 60.0

    # Noise
    noise_object = RelaxationNoise(t1=20.0)
    processor.add_noise(noise_object)
    assert noise_object in processor.noise
    with pytest.raises(TypeError):
        processor.add_noise("non-noise-object")


def test_change_parameters_in_processor():
    processor = LinearSpinChain(0, sx=0.1)
    assert(all(processor.params["sx"] == [0.1]))
