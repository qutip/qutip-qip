import os

from numpy.testing import (assert_, run_module_suite, assert_allclose,
                           assert_equal)
import numpy as np
import pytest

from qutip_qip.device import OptPulseProcessor, SpinChainModel
from qutip_qip.circuit import QubitCircuit
from qutip_qip.qubits import qubit_states
import qutip
from qutip import (fidelity, Qobj, tensor, Options,rand_ket, basis,  sigmaz,
                    sigmax, sigmay, identity, destroy)
from qutip_qip.operations import (cnot, gate_sequence_product,
                                        hadamard_transform, expand_operator)


class TestOptPulseProcessor:
    def test_simple_hadamard(self):
        """
        Test for optimizing a simple hadamard gate
        """
        N = 1
        H_d = sigmaz()
        H_c = sigmax()
        qc = QubitCircuit(N)
        qc.add_gate("SNOT", 0)

        # test load_circuit, with verbose info
        num_tslots = 10
        evo_time = 10
        test = OptPulseProcessor(N, drift=H_d)
        test.add_control(H_c, targets=0)
        tlist, coeffs = test.load_circuit(
            qc, num_tslots=num_tslots, evo_time=evo_time, verbose=True)

        # test run_state
        rho0 = qubit_states(1, [0])
        plus = (qubit_states(1, [0]) + qubit_states(1, [1])).unit()
        result = test.run_state(rho0)
        assert_allclose(fidelity(result.states[-1], plus), 1, rtol=1.0e-6)

    def test_multi_qubits(self):
        """
        Test for multi-qubits system.
        """
        N = 3
        H_d = tensor([sigmaz()]*3)
        H_c = []

        # test empty ctrls
        num_tslots = 30
        evo_time = 10
        test = OptPulseProcessor(N)
        test.add_drift(H_d, [0, 1, 2])
        test.add_control(tensor([sigmax(), sigmax()]),
                          cyclic_permutation=True)
        # test periodically adding ctrls
        sx = sigmax()
        iden = identity(2)
        assert(len(test.get_control_labels()) == 3)
        test.add_control(sigmax(), cyclic_permutation=True)
        test.add_control(sigmay(), cyclic_permutation=True)

        # test pulse genration for cnot gate, with kwargs
        qc = [tensor([identity(2), cnot()])]
        test.load_circuit(qc, num_tslots=num_tslots,
                          evo_time=evo_time, min_fid_err=1.0e-6)
        rho0 = qubit_states(3, [1, 1, 1])
        rho1 = qubit_states(3, [1, 1, 0])
        result = test.run_state(
            rho0, options=Options(store_states=True))
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

    def test_multi_gates(self):
        N = 2
        H_d = tensor([sigmaz()]*2)
        H_c = []

        test = OptPulseProcessor(N)
        test.add_drift(H_d, [0, 1])
        test.add_control(sigmax(), cyclic_permutation=True)
        test.add_control(sigmay(), cyclic_permutation=True)
        test.add_control(tensor([sigmay(), sigmay()]))

        # qubits circuit with 3 gates
        setting_args = {"SNOT": {"num_tslots": 10, "evo_time": 1},
                        "SWAP": {"num_tslots": 30, "evo_time": 3},
                        "CNOT": {"num_tslots": 30, "evo_time": 3}}
        qc = QubitCircuit(N)
        qc.add_gate("SNOT", 0)
        qc.add_gate("SWAP", targets=[0, 1])
        qc.add_gate('CNOT', controls=1, targets=[0])
        test.load_circuit(qc, setting_args=setting_args,
                          merge_gates=False)

        rho0 = rand_ket(4)  # use random generated ket state
        rho0.dims = [[2, 2], [1, 1]]
        U = gate_sequence_product(qc.propagators())
        rho1 = U * rho0
        result = test.run_state(rho0)
        assert_(fidelity(result.states[-1], rho1) > 1-1.0e-6)

    def test_with_model(self):
        model = SpinChainModel(3, setup="linear")
        processor = OptPulseProcessor(3, model=model)
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("X", 2)
        processor.load_circuit(
            qc, merge_gates=True, num_tslots=10, evo_time=2.0
        )
        init_state = qutip.rand_ket(8, dims=[[2, 2, 2], [1, 1, 1]])
        num_result = processor.run_state(init_state=init_state).states[-1]
        ideal_result = qc.run(init_state)
        assert (
            pytest.approx(qutip.fidelity(num_result, ideal_result), 1.0e-6)
            == 1.0
        )
