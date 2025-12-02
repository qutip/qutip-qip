import os

from numpy.testing import (
    assert_allclose, assert_equal)
import numpy as np
import pytest

import qutip
from qutip_qip.device import Processor, LinearSpinChain
from qutip import (
    basis, sigmaz, sigmax, identity, destroy, tensor,
    rand_ket, rand_dm, fidelity)
from qutip_qip.operations import hadamard_transform
from qutip_qip.noise import (
    DecoherenceNoise, RandomNoise, ControlAmpNoise)
from qutip_qip.qubits import qubit_states
from qutip_qip.pulse import Pulse
from qutip_qip.circuit import QubitCircuit


class TestCircuitProcessor:
    def test_control_and_coeffs(self):
        processor = Processor(2)
        processor.add_control(sigmax())
        processor.add_control(sigmaz())

        # Set coeffs and tlist without a label
        coeffs = np.array([[1., 2., 3.], [3., 2., 1.]])
        processor.set_coeffs(coeffs)
        assert_allclose(coeffs, processor.coeffs)
        tlist = np.array([0., 1., 2., 3.])
        processor.set_tlist(tlist)
        assert_allclose(tlist, processor.get_full_tlist())
        processor.set_tlist({0: tlist, 1: tlist})
        assert_allclose(tlist, processor.get_full_tlist())

        # Pulses
        assert(len(processor.pulses) == 2)
        assert(processor.find_pulse(0) == processor.pulses[0])
        assert(processor.find_pulse(1) == processor.pulses[1])
        with pytest.raises(KeyError):
            processor.find_pulse("non_exist_pulse")

    def test_save_read(self):
        """
        Test for saving and reading a pulse matrix
        """
        proc = Processor(num_qubits=2)
        proc.add_control(sigmaz(), label="sz")
        proc.add_control(sigmax(), label="sx")
        proc1 = Processor(num_qubits=2)
        proc1.add_control(sigmaz(), label="sz")
        proc1.add_control(sigmax(), label="sx")
        proc2 = Processor(num_qubits=2)
        proc2.add_control(sigmaz(), label="sz")
        proc2.add_control(sigmax(), label="sx")
        # TODO generalize to different tlist
        tlist = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        amp1 = np.arange(0, 5, 1)
        amp2 = np.arange(5, 0, -1)

        proc.set_all_coeffs(
            {
                label: amp
                for label, amp in zip(proc.get_control_labels(), [amp1, amp2])
            }
        )
        proc.set_all_tlist(tlist)
        proc.save_coeff("qutip_test_CircuitProcessor.txt")
        proc1.read_coeff("qutip_test_CircuitProcessor.txt")
        os.remove("qutip_test_CircuitProcessor.txt")
        assert_allclose(proc1.get_full_coeffs(), proc.get_full_coeffs())
        assert_allclose(proc1.get_full_tlist(), proc.get_full_tlist())
        proc.save_coeff("qutip_test_CircuitProcessor.txt", inctime=False)
        proc2.read_coeff("qutip_test_CircuitProcessor.txt", inctime=False)
        proc2.set_all_tlist(tlist)
        os.remove("qutip_test_CircuitProcessor.txt")
        assert_allclose(proc2.get_full_coeffs(), proc.get_full_coeffs())

    def test_id_evolution(self):
        """
        Test for identity evolution
        """
        N = 1
        proc = Processor(num_qubits=N)
        init_state = rand_ket(2)
        tlist = [0., 1., 2.]
        proc.add_pulse(Pulse(identity(2), 0, tlist, False))
        result = proc.run_state(
            init_state, options={'store_final_state': True})
        global_phase = init_state[0, 0]/result.final_state[0, 0]
        assert_allclose(
            global_phase*result.final_state.full(), init_state.full())

    def test_id_with_T1_T2(self):
        """
        Test for identity evolution with relaxation t1 and t2
        """
        # setup
        a = destroy(2)
        Hadamard = hadamard_transform(1)
        ex_state = basis(2, 1)
        mines_state = (basis(2, 1)-basis(2, 0)).unit()
        end_time = 2.
        tlist = np.arange(0, end_time + 0.02, 0.02)
        t1 = 1.
        t2 = 0.5

        # test t1
        test = Processor(1, t1=t1)
        # zero ham evolution
        test.add_pulse(Pulse(identity(2), 0, tlist, False))
        result = test.run_state(ex_state, e_ops=[a.dag()*a])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./t1*end_time),
            rtol=1e-5, err_msg="Error in t1 time simulation")

        # test t2
        test = Processor(1, t2=t2)
        test.add_pulse(Pulse(identity(2), 0, tlist, False))
        result = test.run_state(
            init_state=mines_state, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./t2*end_time)*0.5+0.5,
            rtol=1e-5, err_msg="Error in t2 time simulation")

        # test t1 and t2
        t1 = np.random.rand(1) + 0.5
        t2 = np.random.rand(1) * 0.5 + 0.5
        test = Processor(1, t1=t1, t2=t2)
        test.add_pulse(Pulse(identity(2), 0, tlist, False))
        result = test.run_state(
            init_state=mines_state, e_ops=[Hadamard*a.dag()*a*Hadamard])
        assert_allclose(
            result.expect[0][-1], np.exp(-1./t2*end_time)*0.5+0.5,
            rtol=1e-5,
            err_msg="Error in t1 & t2 simulation, "
                    "with t1={} and t2={}".format(t1, t2))

    def test_plot(self):
        """
        Test for plotting functions
        """
        plt = pytest.importorskip("matplotlib.pyplot")

        # step_func
        tlist = np.linspace(0., 2*np.pi, 20)
        processor = Processor(num_qubits=1, spline_kind="step_func")
        processor.add_control(sigmaz(), label="sz")
        processor.set_all_coeffs({"sz": np.array([np.sin(t) for t in tlist])})
        processor.set_all_tlist(tlist)
        fig, _ = processor.plot_pulses(use_control_latex=False)
        # testing under Xvfb with pytest-xvfb complains if figure windows are
        # left open, so we politely close it:
        plt.close(fig)

        # cubic spline
        tlist = np.linspace(0., 2*np.pi, 20)
        processor = Processor(num_qubits=1, spline_kind="cubic")
        processor.add_control(sigmaz(), label="sz")
        processor.set_all_coeffs({"sz": np.array([np.sin(t) for t in tlist])})
        processor.set_all_tlist(tlist)
        fig, _ = processor.plot_pulses(use_control_latex=False)
        # testing under Xvfb with pytest-xvfb complains if figure windows are
        # left open, so we politely close it:
        plt.close(fig)

    def testSpline(self):
        """
        Test if the spline kind is correctly transferred into
        the arguments in QobjEvo.
        """
        # We use a varying coefficient to distinctively test 
        # step_func vs cubic interpolation behavior.
        tlist = np.array([0, 1, 2, 3], dtype=float)
        coeff = np.array([0, 1, 2, 3], dtype=float)
        
        processor = Processor(num_qubits=1, spline_kind="step_func")
        processor.add_control(sigmaz(), label="sz")
        processor.set_all_coeffs({"sz": coeff})
        processor.set_tlist(tlist)

        ideal_qobjevo, _ = processor.get_qobjevo(noisy=False)
        assert_allclose(ideal_qobjevo(0.5).data.to_array(), sigmaz().data.to_array() * 0.0)

        # Verify noise inherits the step property
        noisy_qobjevo, c_ops = processor.get_qobjevo(noisy=True)
        assert_allclose(noisy_qobjevo(0.5).data.to_array(), sigmaz().data.to_array() * 0.0)

        processor_cubic = Processor(num_qubits=1, spline_kind="cubic")
        processor_cubic.add_control(sigmaz(), label="sz")
        processor_cubic.set_all_coeffs({"sz": coeff})
        processor_cubic.set_all_tlist(tlist)

        ideal_qobjevo_cubic, _ = processor_cubic.get_qobjevo(noisy=False)
        val_at_half = ideal_qobjevo_cubic(0.5).data.to_array()[0,0]
        assert abs(val_at_half - 0.5) < 0.1 
        assert abs(val_at_half - 0.0) > 0.1

    def testGetObjevo(self):
        """
        Test structure of QobjEvo (Adapted for QuTiP v5).
        """
        tlist = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        coeff = np.array([1, 1, 1, 1, 1, 1], dtype=float)
        processor = Processor(num_qubits=1)
        processor.add_control(sigmaz(), label="sz")
        processor.set_all_coeffs({"sz": coeff})
        processor.set_all_tlist(tlist)

        unitary_qobjevo, _ = processor.get_qobjevo(args={"test": True}, noisy=False)
        components = unitary_qobjevo.to_list()
        assert_allclose(
            components[0][0].data.to_array(), 
            sigmaz().data.to_array()
        )
        
        # For v5: QobjEvo has no public .tlist attribute.
        # Instead, verify the time-dependence works by evaluating it at t=1.0.
        assert_allclose(unitary_qobjevo(1.0).data.to_array(), sigmaz().data.to_array())
        
        # With Decoherence Noise
        dec_noise = DecoherenceNoise(
            c_ops=sigmax(), coeff=coeff, tlist=tlist)
        processor.add_noise(dec_noise)
        
        assert_equal(unitary_qobjevo.to_list()[0][0],
                     processor.get_qobjevo(noisy=False)[0].to_list()[0][0]) #We index [0] to get the Hamiltonian before calling .to_list()

        # Check that sigmaz is present in the Hamiltonian components with noise
        noisy_qobjevo, c_ops = processor.get_qobjevo(args={"test": True}, noisy=True)
        hamiltonian_parts = [pair[0] for pair in noisy_qobjevo.to_list()]
        assert any(pair == sigmaz() for pair in hamiltonian_parts)

        # Check collapse operators - c_ops[0] is a QobjEvo. We inspect its components.
        c_op_comp = c_ops[0].to_list()
        assert_equal(c_op_comp[0][0], sigmax())
        
        # For v5: Verify value at t=1.0 instead of checking .tlist attribute
        assert_allclose(c_ops[0](1.0).data.to_array(), sigmax().data.to_array())

        # With Amplitude Noise
        processor = Processor(num_qubits=1, spline_kind="cubic")
        processor.add_control(sigmaz(), label="sz")
        tlist = np.linspace(1, 6, int(5/0.2))
        
        # Use random coeff to verify values explicitly
        coeff = np.random.rand(len(tlist))
        processor.set_all_coeffs({"sz": coeff})
        processor.set_all_tlist(tlist)

        amp_noise = ControlAmpNoise(coeff=coeff, tlist=tlist)
        processor.add_noise(amp_noise)
        
        noisy_qobjevo, c_ops = processor.get_qobjevo(
            args={"test": True}, noisy=True)

        # We verify the total physical value: (Signal + Noise).
        # At t = tlist[0], the total coefficient should be:
        # Signal (coeff[0]) + Noise (coeff[0]) = 2 * coeff[0]
        t_check = tlist[0]
        expected_coeff_val = coeff[0] + coeff[0]
        qobj_at_t = noisy_qobjevo(t_check)
        
        # Extract the scalar value at index [0,0] (corresponding to sigmaz diagonal)
        # sigmaz is diag(1, -1), so element [0,0] is +1 * coefficient
        actual_val = qobj_at_t.data.to_array()[0, 0]
        assert_allclose(actual_val, expected_coeff_val, rtol=1.e-5)
        
        # Verify the operator structure is still sigmaz
        # If we divide by the expected coefficient, we should get exactly sigmaz
        if abs(expected_coeff_val) > 1e-9:
            normalized_qobj = qobj_at_t / expected_coeff_val
            assert_allclose(normalized_qobj.data.to_array(), sigmaz().data.to_array(), atol=1e-5)

    def testNoise(self):
        """
        Test for Processor with noise
        """
        # setup and fidelity without noise
        init_state = qubit_states(2, [0, 0, 0, 0])
        tlist = np.array([0., np.pi/2.])
        a = destroy(2)
        proc = Processor(num_qubits=2)
        proc.add_control(sigmax(), targets=1, label="sx")
        proc.set_all_coeffs({"sx": np.array([1.])})
        proc.set_all_tlist(tlist)
        result = proc.run_state(init_state=init_state)
        assert_allclose(
            fidelity(
                result.states[-1],
                qubit_states(2, [0, 1, 0, 0])
            ),
            1, rtol=1.e-7
        )

        # decoherence noise
        dec_noise = DecoherenceNoise([0.25*a], targets=1)
        proc.add_noise(dec_noise)
        result = proc.run_state(init_state=init_state)
        assert_allclose(
            fidelity(
                result.states[-1],
                qubit_states(2, [0, 1, 0, 0])
            ),
            0.981852, rtol=1.e-3
        )

        # white random noise
        proc.model._noise = []
        white_noise = RandomNoise(0.2, np.random.normal, loc=0.1, scale=0.1)
        proc.add_noise(white_noise)
        result = proc.run_state(init_state=init_state)

    def testMultiLevelSystem(self):
        """
        Test for processor with multi-level system
        """
        N = 2
        proc = Processor(num_qubits=N, dims=[2, 3])
        proc.add_control(tensor(sigmaz(), rand_dm(3, density=1.)), label="sz0")
        proc.set_all_coeffs({"sz0": np.array([1, 2])})
        proc.set_all_tlist(np.array([0., 1., 2]))
        proc.run_state(init_state=tensor([basis(2, 0), basis(3, 1)]))

    def testDrift(self):
        """
        Test for the drift Hamiltonian
        """
        processor = Processor(num_qubits=1)
        processor.add_drift(sigmax() / 2, 0)
        tlist = np.array([0., np.pi, 2*np.pi, 3*np.pi])
        processor.add_pulse(Pulse(None, None, tlist, False))
        ideal_qobjevo, _ = processor.get_qobjevo(noisy=True)
        assert_equal(ideal_qobjevo(0), sigmax() / 2)

        init_state = basis(2)
        propagators = processor.run_analytically()
        analytical_result = init_state
        for unitary in propagators:
            analytical_result = unitary * analytical_result
        fid = fidelity(sigmax() * init_state, analytical_result)
        assert((1 - fid) < 1.0e-6)

    def testChooseSolver(self):
        # setup and fidelity without noise
        init_state = qubit_states(2, [0, 0, 0, 0])
        tlist = np.linspace(0., np.pi/2., 10)
        a = destroy(2)
        proc = Processor(num_qubits=2, t2=100)
        proc.add_control(sigmax(), targets=1, label="sx")
        proc.set_all_coeffs({"sx": np.array([1.] * len(tlist))})
        proc.set_all_tlist(tlist)
        observerable = tensor([qutip.qeye(2), qutip.sigmax()])
        result1 = proc.run_state(
            init_state=init_state, solver="mcsolve", e_ops=observerable)
        assert result1.solver == "mcsolve"

    def test_no_saving_intermidiate_state(self):
        processor = Processor(1)
        processor.add_pulse(pulse=
            Pulse(sigmax(), coeff=np.ones(10),
            tlist=np.linspace(0,1,10), targets=0)
            )
        result = processor.run_state(basis(2,0), tlist=[0,1])
        assert(len(result.states) == 2)

    def test_repeated_use_of_processor(self):
        processor = Processor(1, t1=1.)
        processor.add_pulse(
            Pulse(sigmax(), targets=0, coeff=True))
        result1 = processor.run_state(basis(2, 0), tlist=np.linspace(0, 1, 10))
        result2 = processor.run_state(basis(2, 0), tlist=np.linspace(0, 1, 10))
        assert_allclose(result1.states[-1].full(), result2.states[-1].full())

    def test_pulse_mode(self):
        processor = Processor(2)
        processor.add_control(sigmax(), targets=0, label="sx")
        processor.set_coeffs({"sx": np.array([1., 2., 3.])})
        processor.set_tlist({"sx": np.array([0., 1., 2., 3.])})

        processor.pulse_mode = "continuous"
        assert(processor.pulse_mode == "continuous")
        assert(processor.pulses[0].spline_kind == "cubic")
        processor.pulse_mode = "discrete"
        assert(processor.pulse_mode == "discrete")
        assert(processor.pulses[0].spline_kind == "step_func")

    def test_max_step_size(self):
        num_qubits = 2
        init_state = tensor([basis(2, 1), basis(2, 1)])
        qc = QubitCircuit(2)

        # ISWAP acts trivially on the initial states.
        # If no max_step are defined,
        # the solver will choose a step size too large
        # such that the X gate will be skipped.
        qc.add_gate("ISWAP", targets=[0, 1])
        qc.add_gate("ISWAP", targets=[0, 1])
        qc.add_gate("X", targets=[0])
        processor = LinearSpinChain(num_qubits)
        processor.load_circuit(qc)

        # No max_step
        final_state = processor.run_state(
            init_state,
            options={'max_step': 10000} # too large max_step
        ).states[-1]
        expected_state = tensor([basis(2, 0), basis(2, 1)])
        assert pytest.approx(fidelity(final_state, expected_state), 0.001) == 0

        # With default max_step
        final_state = processor.run_state(init_state).states[-1]
        expected_state = tensor([basis(2, 0), basis(2, 1)])
        assert pytest.approx(fidelity(final_state, expected_state), 0.001) == 1
