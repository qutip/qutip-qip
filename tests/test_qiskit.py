import pytest
import sys
import numpy as np
from qutip_qip.circuit import QubitCircuit
from qiskit import QuantumCircuit

from qutip_qip.qiskit.converter import qiskit_to_qutip, _get_qutip_index


def _check_for_qiskit() -> bool:
    """
    Check whether qiskit is installed.
    """
    return True if "qiskit" in sys.modules else False


class TestConverter:
    """
    Class for testing only the circuit conversion
    from qiskit to qutip.
    """

    def _compare_circuit(
        self, result_circuit: QubitCircuit, required_circuit: QubitCircuit
    ) -> bool:
        """
        Check whether two circuits are equivalent.
        """
        # to be corrected
        if result_circuit.N != required_circuit.N or len(
            result_circuit.gates
        ) != len(required_circuit.gates):
            return False

        for i, res_gate in enumerate(result_circuit.gates):
            req_gate = required_circuit.gates[i]

            check_condition = (req_gate.name == res_gate.name) and (
                req_gate.targets
                == [
                    _get_qutip_index(target, result_circuit.N)
                    for target in res_gate.targets
                ]
            )
            if not check_condition:
                return False

            if req_gate.name == "measure":
                check_condition = (
                    req_gate.classical_store
                    == res_gate._get_qutip_index(
                        res_gate.classical_store, result_circuit.num_cbits
                    )
                )
            else:
                # todo: correct for float error in arg_value
                res_arg = res_gate.arg_value if res_gate.arg_value else None
                req_arg = req_gate.arg_value if req_gate.arg_value else None
                res_controls = (
                    [
                        _get_qutip_index(control, result_circuit.N)
                        for control in res_gate.controls
                    ]
                    if res_gate.controls
                    else None
                )
                req_controls = req_gate.controls if req_gate.controls else None

                check_condition = (res_arg == req_arg) and (
                    res_controls == req_controls
                )

        return check_condition

    def test_single_qubit_conversion(self):
        """
        Test to check conversion of a circuit
        containing a single qubit gate.
        """
        if not _check_for_qiskit():
            return

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.x(0)
        result_circuit = qiskit_to_qutip(qiskit_circuit)
        required_circuit = QubitCircuit(1)
        required_circuit.add_gate("X", targets=[0])

        assert self._compare_circuit(result_circuit, required_circuit)

    def test_controlled_qubit_conversion(self):
        """
        Test to check conversion of a circuit
        containing a 2 qubit controlled gate.
        """
        if not _check_for_qiskit():
            return
        qiskit_circuit = QuantumCircuit(2)
        qiskit_circuit.cx(1, 0)
        result_circuit = qiskit_to_qutip(qiskit_circuit)

        required_circuit = QubitCircuit(2)
        required_circuit.add_gate("CX", targets=[0], controls=[1])

        assert self._compare_circuit(result_circuit, required_circuit)

    def test_rotation_conversion(self):
        """
        Test to check conversion of a circuit
        containing a single qubit rotation gate.
        """
        if not _check_for_qiskit():
            return

        qiskit_circuit = QuantumCircuit(1)
        qiskit_circuit.rx(np.pi / 3, 0)
        result_circuit = qiskit_to_qutip(qiskit_circuit)
        required_circuit = QubitCircuit(1)
        required_circuit.add_gate("RX", targets=[0], arg_value=np.pi / 3)

        assert self._compare_circuit(result_circuit, required_circuit)
