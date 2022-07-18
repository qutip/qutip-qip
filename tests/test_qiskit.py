import pytest
import sys
import numpy as np
from yaml import parse
from qutip_qip.circuit import QubitCircuit
from qiskit import QuantumCircuit

from qutip_qip.qiskit.converter import qiskit_to_qutip, _get_qutip_index


def _check_for_qiskit():
    return True if "qiskit" in sys.modules else False


class TestConverter:

    def _parse_circuit(self, circuit):
        parsed_circuit = []
        for gate in circuit.gates:
            name = gate.name
            targets = [_get_qutip_index(target, circuit.N)
                       for target in gate.targets]
            if name == "measure":
                classical_store = gate._get_qutip_index(
                    classical_store, circuit.num_cbits)
                parsed_circuit.append((name, targets, classical_store))
            else:
                arg_value = gate.arg_value if gate.arg_value else None
                controls = [_get_qutip_index(
                    control, circuit.N) for control in gate.controls] if gate.controls else None
                parsed_circuit.append((name, targets, controls, arg_value))
        return parsed_circuit

    def _compare_circuit(self, result_circuit, required_circuit):
        # to be corrected
        if result_circuit.N != required_circuit.N:
            return False

        parsed_result = self._parse_circuit(result_circuit)
        parsed_required = self._parse_circuit(required_circuit)

        for i, gate in enumerate(parsed_required):
            if gate in parsed_result:
                parsed_result.remove(gate)
            else:
                return False

        return True if len(parsed_result) == 0 else False

    def test_single_qubit_conversion(self):
        if not _check_for_qiskit():
            return

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.x(0)
        result_circuit = qiskit_to_qutip(qiskit_circuit)
        required_circuit = QubitCircuit(1)
        required_circuit.add_gate("X", targets=[0])

        assert(self._compare_circuit(result_circuit, required_circuit))

    def test_controlled_qubit_conversion(self):
        if not _check_for_qiskit():
            return

        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.cx(0, 1)
        result_circuit = qiskit_to_qutip(qiskit_circuit)
        required_circuit = QubitCircuit(2)
        required_circuit.add_gate("CX", targets=[0], controls=[1])

        assert(self._compare_circuit(result_circuit, required_circuit))

    # def test_rotation_conversion(self):
    #     if not _check_for_qiskit():
    #         return

    #     qiskit_circuit = QuantumCircuit(1)
    #     qiskit_circuit.rx(np.pi/3, 0)
    #     result_circuit = qiskit_to_qutip(qiskit_circuit)
    #     required_circuit = QubitCircuit(1)
    #     required_circuit.add_gate(
    #         "RX", targets=[0], arg_value=np.pi/3)

    #     assert(self._compare_circuit(result_circuit, required_circuit))
