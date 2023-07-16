from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest

import qutip
from qutip import (
    Qobj, sigmax, sigmay, sigmaz, identity, tensor, QobjEvo
)
from qutip_qip.pulse import Pulse, Drift


def _compare_qobjevo(qevo1, qevo2, t_min, t_max):
    for t in t_min + np.random.rand(25) * (t_max - t_min):
        assert_allclose(qevo1(t).full(), qevo2(t).full())


class TestPulse:
    def test_basic_pulse(self):
        """
        Test for basic pulse generation and attributes.
        """
        coeff = np.array([0.1, 0.2, 0.3, 0.4])
        tlist = np.array([0., 1., 2., 3.])
        ham = sigmaz()

        # Basic tests
        pulse1 = Pulse(ham, 1, tlist, coeff)
        assert_allclose(
            pulse1.get_ideal_qobjevo(2)(0).full(),
            tensor(identity(2), sigmaz()).full() * coeff[0])
        pulse1.tlist = 2 * tlist
        assert_allclose(pulse1.tlist, 2 * tlist)
        pulse1.tlist = tlist
        pulse1.coeff = 2 * coeff
        assert_allclose(pulse1.coeff, 2 * coeff)
        pulse1.coeff = coeff
        pulse1.qobj = 2 * sigmay()
        assert_allclose(pulse1.qobj.full(), 2 * sigmay().full())
        pulse1.qobj = ham
        pulse1.targets = 3
        assert_allclose(pulse1.targets, 3)
        pulse1.targets = 1
        qobjevo = pulse1.get_ideal_qobjevo(2)
        if parse_version(qutip.__version__) >= parse_version('5.dev'):
            expected = QobjEvo(
                [tensor(identity(2), sigmaz()), coeff],
                tlist=tlist,
                order=0
                )
        else:
            expected = QobjEvo(
                [tensor(identity(2), sigmaz()), coeff],
                tlist=tlist,
                args={"_step_func_coeff": True}
                )
        _compare_qobjevo(qobjevo, expected, 0, 3)

    def test_coherent_noise(self):
        """
        Test for pulse genration with coherent noise.
        """
        coeff = np.array([0.1, 0.2, 0.3, 0.4])
        tlist = np.array([0., 1., 2., 3.])
        ham = sigmaz()
        pulse1 = Pulse(ham, 1, tlist, coeff)
        # Add coherent noise with the same tlist
        pulse1.add_coherent_noise(sigmay(), 0, tlist, coeff)
        assert_allclose(
            pulse1.get_ideal_qobjevo(2)(0).full(),
            tensor(identity(2), sigmaz()).full() * 0.1)
        assert_(len(pulse1.coherent_noise) == 1)
        noise_qu, c_ops = pulse1.get_noisy_qobjevo(2)
        assert_allclose(c_ops, [])
        assert_allclose(pulse1.get_full_tlist(), np.array([0., 1., 2., 3.]))

    def test_noisy_pulse(self):
        """
        Test for lindblad noise and different tlist
        """
        coeff = np.array([0.1, 0.2, 0.3, 0.4])
        tlist = np.array([0., 1., 2., 3.])
        ham = sigmaz()
        pulse1 = Pulse(ham, 1, tlist, coeff)
        # Add coherent noise and lindblad noise with different tlist
        pulse1.spline_kind = "step_func"
        tlist_noise = np.array([0., 1., 2.5, 3.])
        coeff_noise = np.array([0., 0.5, 0.1, 0.5])
        pulse1.add_coherent_noise(sigmay(), 0, tlist_noise, coeff_noise)
        tlist_noise2 = np.array([0., 0.5, 2, 3.])
        coeff_noise2 = np.array([0., 0.1, 0.2, 0.3])
        pulse1.add_lindblad_noise(sigmax(), 1, coeff=True)
        pulse1.add_lindblad_noise(
            sigmax(), 0, tlist=tlist_noise2, coeff=coeff_noise2)

        assert_allclose(
            pulse1.get_ideal_qobjevo(2)(0).full(),
            tensor(identity(2), sigmaz()).full() * 0.1)
        noise_qu, c_ops = pulse1.get_noisy_qobjevo(2)
        assert_allclose(
            pulse1.get_full_tlist(), np.array([0., 0.5,  1., 2., 2.5, 3.]))
        if parse_version(qutip.__version__) >= parse_version('5.dev'):
            expected = QobjEvo([
                    [tensor(identity(2), sigmaz()),
                        np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4])],
                    [tensor(sigmay(), identity(2)),
                        np.array([0., 0., 0.5, 0.5, 0.1, 0.5])]
                ],
                tlist=np.array([0., 0.5,  1., 2., 2.5, 3.]),
                order=0)
        else:
            expected = QobjEvo([
                    [tensor(identity(2), sigmaz()),
                        np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4])],
                    [tensor(sigmay(), identity(2)),
                        np.array([0., 0., 0.5, 0.5, 0.1, 0.5])]
                ],
                tlist=np.array([0., 0.5,  1., 2., 2.5, 3.]),
                args={"_step_func_coeff": True})
        _compare_qobjevo(noise_qu, expected, 0, 3)

        for c_op in c_ops:
            try:
                isconstant = c_op.isconstant
            except AttributeError:
                isconstant = (len(c_op.ops) == 0)
            if isconstant:
                assert_allclose(c_op(0).full(),
                                tensor(identity(2), sigmax()).full())
            else:
                if parse_version(qutip.__version__) >= parse_version('5.dev'):
                    expected = QobjEvo(
                        [tensor(sigmax(), identity(2)),
                            np.array([0., 0.1, 0.1, 0.2, 0.2, 0.3])],
                        tlist=np.array([0., 0.5,  1., 2., 2.5, 3.]),
                        order=0)
                else:
                    expected = QobjEvo(
                        [tensor(sigmax(), identity(2)),
                            np.array([0., 0.1, 0.1, 0.2, 0.2, 0.3])],
                        tlist=np.array([0., 0.5,  1., 2., 2.5, 3.]),
                        args={"_step_func_coeff": True})
                _compare_qobjevo(c_op, expected, 0, 3)

    def test_pulse_constructor(self):
        """
        Test for creating empty Pulse, Pulse with constant coefficients etc.
        """
        coeff = np.array([0.1, 0.2, 0.3, 0.4])
        tlist = np.array([0., 1., 2., 3.])
        ham = sigmaz()
        # Special ways of initializing pulse
        pulse2 = Pulse(sigmax(), 0, tlist, True)
        assert_allclose(pulse2.get_ideal_qobjevo(2)(0).full(),
                        tensor(sigmax(), identity(2)).full())

        pulse3 = Pulse(sigmay(), 0)
        assert_allclose(pulse3.get_ideal_qobjevo(2)(0).norm(), 0.)

        pulse4 = Pulse(None, None)  # Dummy empty ham
        assert_allclose(pulse4.get_ideal_qobjevo(2)(0).norm(), 0.)

        tlist_noise = np.array([1., 2.5, 3.])
        coeff_noise = np.array([0.5, 0.1, 0.5])
        tlist_noise2 = np.array([0.5, 2, 3.])
        coeff_noise2 = np.array([0.1, 0.2, 0.3])
        # Pulse with different dims
        random_qobj = Qobj(np.random.random((3, 3)))
        pulse5 = Pulse(sigmaz(), 1, tlist, True)
        pulse5.add_coherent_noise(sigmay(), 1, tlist_noise, coeff_noise)
        pulse5.add_lindblad_noise(
            random_qobj, 0, tlist=tlist_noise2, coeff=coeff_noise2)
        qu, c_ops = pulse5.get_noisy_qobjevo(dims=[3, 2])
        if parse_version(qutip.__version__) >= parse_version('5.dev'):
            expected = QobjEvo(
                [
                    tensor([identity(3), sigmaz()]),
                    [tensor([identity(3), sigmay()]), coeff_noise]
                ],
                tlist=tlist_noise,
                order=0)
        else:
            expected = QobjEvo(
                [
                    tensor([identity(3), sigmaz()]),
                    [tensor([identity(3), sigmay()]), coeff_noise]
                ],
                tlist=tlist_noise,
                args={"_step_func_coeff": True})
        _compare_qobjevo(qu, expected, 0, 3)

    def test_drift(self):
        """
        Test for Drift
        """
        drift = Drift()
        assert_allclose(drift.get_ideal_qobjevo(2)(0).norm(), 0)
        drift.add_drift(sigmaz(), targets=1)
        assert_allclose(
            drift.get_ideal_qobjevo(dims=[3, 2])(0).full(),
            tensor(identity(3), sigmaz()).full())
