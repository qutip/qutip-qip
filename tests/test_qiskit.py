import pytest
import numpy as np
import random
from numpy.testing import assert_allclose
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import (
    LinearSpinChain,
    CircularSpinChain,
    DispersiveCavityQED,
)

# will skip tests in this entire file
# if qiskit is not installed
pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qutip_qip.qiskit.provider import (
    QiskitCircuitSimulator,
    QiskitPulseSimulator,
)
from qutip_qip.qiskit.converter import (
    convert_qiskit_circuit,
    _get_qutip_index,
)


class TestConverter:
    """
    Class for testing only the circuit conversion
    from qiskit to qutip.
    """

    def _compare_args(self, req_gate, res_gate):
        """Compare parameters of two gates"""
        res_arg = (
            (
                res_gate.arg_value
                if type(res_gate.arg_value) is list
                or type(res_gate.arg_value) is tuple
                else [res_gate.arg_value]
            )
            if res_gate.arg_value
            else []
        )

        req_arg = (
            (
                req_gate.arg_value
                if type(req_gate.arg_value) is list
                or type(req_gate.arg_value) is tuple
                else [req_gate.arg_value]
            )
            if req_gate.arg_value
            else []
        )

        if len(req_arg) != len(res_arg):
            return False

        return np.allclose(req_arg, res_arg)

    def _compare_gate(self, req_gate, res_gate, result_circuit: QubitCircuit):
        """Check whether two gates are equivalent"""
        check_condition = (req_gate.name == res_gate.name) and (
            req_gate.targets
            == _get_qutip_index(res_gate.targets, result_circuit.N)
        )
        if not check_condition:
            return False

        if req_gate.name == "measure":
            check_condition = req_gate.classical_store == _get_qutip_index(
                res_gate.classical_store, result_circuit.num_cbits
            )
        else:
            # todo: correct for float error in arg_value
            res_controls = (
                _get_qutip_index(res_gate.controls, result_circuit.N)
                if res_gate.controls
                else None
            )
            req_controls = req_gate.controls if req_gate.controls else None

            check_condition = (
                res_controls == req_controls
            ) and self._compare_args(req_gate, res_gate)

        return check_condition

    def _compare_circuit(
        self, result_circuit: QubitCircuit, required_circuit: QubitCircuit
    ) -> bool:
        """
        Check whether two circuits are equivalent.
        """
        if result_circuit.N != required_circuit.N or len(
            result_circuit.gates
        ) != len(required_circuit.gates):
            return False

        for i, res_gate in enumerate(result_circuit.gates):
            req_gate = required_circuit.gates[i]

            if not self._compare_gate(req_gate, res_gate, result_circuit):
                return False

        return True

    def test_single_qubit_conversion(self):
        """
        Test to check conversion of a circuit
        containing a single qubit gate.
        """
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.x(0)
        result_circuit = convert_qiskit_circuit(qiskit_circuit)
        required_circuit = QubitCircuit(1)
        required_circuit.add_gate("X", targets=[0])

        assert self._compare_circuit(result_circuit, required_circuit)

    def test_controlled_qubit_conversion(self):
        """
        Test to check conversion of a circuit
        containing a 2 qubit controlled gate.
        """
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.cx(1, 0)
        result_circuit = convert_qiskit_circuit(qiskit_circuit)

        required_circuit = QubitCircuit(2)
        required_circuit.add_gate("CX", targets=[0], controls=[1])

        assert self._compare_circuit(result_circuit, required_circuit)

    def test_rotation_conversion(self):
        """
        Test to check conversion of a circuit
        containing a single qubit rotation gate.
        """
        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.rx(np.pi / 3, 0)
        result_circuit = convert_qiskit_circuit(qiskit_circuit)
        required_circuit = QubitCircuit(1)
        required_circuit.add_gate("RX", targets=[0], arg_value=np.pi / 3)

        assert self._compare_circuit(result_circuit, required_circuit)


class TestSimulator:
    """
    Class for testing whether a simulator
    gives correct results.
    """

    def test_circuit_simulator(self):
        """
        Test whether the circuit_simulator matches the
        results of qiskit's statevector simulator.
        """

        # test single qubit operations
        circ1 = QuantumCircuit(2, 2)
        circ1.h(0)
        circ1.h(1)
        self._compare_results(circ1)

        # test controlled operations
        circ2 = QuantumCircuit(2, 2)
        circ2.h(0)
        circ2.cx(0, 1)
        self._compare_results(circ2)

    def test_allow_custom_gate(self):
        """
        Asserts whether execution will fail on running a circuit
        with a custom sub-circuit with the option allow_custom_gate=False
        """
        with pytest.raises(RuntimeError):
            circ = QuantumCircuit(2, 2)
            circ.h(0)
            # make a custom sub-circuit
            sub_circ = QuantumCircuit(1)
            sub_circ.x(0)
            my_gate = sub_circ.to_gate()
            circ.append(my_gate, [1])

            qutip_backend = QiskitCircuitSimulator()
            # running this with allow_custom_gate=False should raise
            # a RuntimeError due to the custom sub-circuit
            qutip_backend.run(circ, allow_custom_gate=False)

    def test_measurements(self):
        """
        Tests measurements by setting a predefined seed to
        obtain predetermined results.
        """
        random.seed(1)
        predefined_counts = {"0": 233, "11": 267, "10": 254, "1": 270}

        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.h(1)
        circ.measure(0, 0)
        circ.measure(1, 1)

        qutip_backend = QiskitCircuitSimulator()
        qutip_job = qutip_backend.run(circ)
        qutip_result = qutip_job.result()

        assert qutip_result.get_counts(circ) == predefined_counts

    def test_lsc_simulator(self):
        """
        Test whether the pulse backend based on the LinearSpinChain model
        matches predefined correct results.
        """
        circ, predefined_counts = self._init_pulse_test()

        result = self._run_pulse_processor(LinearSpinChain(num_qubits=2), circ)
        assert result.get_counts() == predefined_counts

    def test_csc_simulator(self):
        """
        Test whether the pulse backend based on the CircularSpinChain model
        matches predefined correct results.
        """
        circ, predefined_counts = self._init_pulse_test()

        result = self._run_pulse_processor(
            CircularSpinChain(num_qubits=2), circ
        )
        assert result.get_counts() == predefined_counts

    def test_cavityqed_simulator(self):
        """
        Test whether the pulse backend based on the DispersiveCavityQED
        model matches predefined correct results.
        """
        circ, predefined_counts = self._init_pulse_test()

        result = self._run_pulse_processor(
            DispersiveCavityQED(num_qubits=2, num_levels=10), circ
        )
        assert result.get_counts() == predefined_counts

    def _compare_results(self, qiskit_circuit: QuantumCircuit):

        qutip_backend = QiskitCircuitSimulator()
        qutip_job = qutip_backend.run(qiskit_circuit)
        qutip_result = qutip_job.result()
        qutip_sv = qutip_result.data()["statevector"]

        qiskit_backend = AerSimulator(method="statevector")
        qiskit_circuit.save_state()
        qiskit_job = qiskit_backend.run(qiskit_circuit)
        qiskit_result = qiskit_job.result()
        qiskit_sv = qiskit_result.data()["statevector"]

        assert_allclose(qutip_sv, qiskit_sv)

    def _run_pulse_processor(self, processor, circ):
        qutip_backend = QiskitPulseSimulator(processor)
        qutip_job = qutip_backend.run(circ)
        return qutip_job.result()

    def _init_pulse_test(self):
        random.seed(1)

        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.h(1)

        predefined_counts = {"0": 233, "11": 267, "1": 254, "10": 270}

        return circ, predefined_counts
