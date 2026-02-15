import numpy as np
from numpy.testing import assert_, assert_equal
import unittest
from qutip import Qobj, sigmaz, tensor
from qutip_qip.operations import controlled_gate_factory, custom_gate_factory

from qutip_qip.algorithms.qpe import qpe


class TestQPE(unittest.TestCase):
    """
    A test class for the Quantum Phase Estimation implementation
    """

    def test_custom_gate(self):
        """
        Test if custom_gate_factory correctly stores and returns the quantum object
        """
        U = Qobj([[0, 1], [1, 0]])

        custom = custom_gate_factory(gate_name="custom", U=U)
        qobj = custom.get_qobj()
        assert_((qobj - U).norm() < 1e-12)

    def test_controlled_unitary(self):
        """
        Test if create_controlled_unitary creates a correct controlled gate
        """
        U = Qobj([[0, 1], [1, 0]])

        controlled_u = controlled_gate_factory(
            gate=custom_gate_factory(gate_name="CU", U=U),
        )(control_value=1)

        assert_equal(controlled_u.control_value, 1)
        assert_(
            (controlled_u.target_gate.get_qobj() - U).norm() < 1e-12
        )

    def test_qpe_validation(self):
        """
        Test input validation in qpe function
        """
        U = sigmaz()

        with self.assertRaises(ValueError):
            qpe(U, num_counting_qubits=0)

        invalid_U = Qobj([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3 matrix
        with self.assertRaises(ValueError):
            qpe(invalid_U, num_counting_qubits=3)

    def test_qpe_circuit_structure(self):
        """
        Test if qpe creates circuit with correct structure
        """
        U = sigmaz()

        num_counting = 3
        circuit = qpe(
            U, num_counting_qubits=num_counting, target_qubits=num_counting
        )

        assert_equal(circuit.num_qubits, num_counting + 1)

        for i in range(num_counting):
            assert_equal(circuit.instructions[i].targets, [i])

        for i in range(num_counting):
            circ_instruction = circuit.instructions[num_counting + i]
            assert_(circ_instruction.operation.is_controlled_gate)
            assert_equal(circ_instruction.controls, [i])
            assert_equal(circ_instruction.targets, [num_counting])

    def test_qpe_different_target_specifications(self):
        """
        Test QPE with different ways of specifying target qubits
        """
        U = sigmaz()
        num_counting = 2

        circuit1 = qpe(
            U, num_counting_qubits=num_counting, target_qubits=num_counting
        )
        assert_equal(circuit1.num_qubits, num_counting + 1)

        circuit2 = qpe(
            U, num_counting_qubits=num_counting, target_qubits=[num_counting]
        )
        assert_equal(circuit2.num_qubits, num_counting + 1)

        circuit3 = qpe(U, num_counting_qubits=num_counting, target_qubits=None)
        assert_equal(circuit3.num_qubits, num_counting + 1)

        U2 = tensor(sigmaz(), sigmaz())
        circuit4 = qpe(
            U2,
            num_counting_qubits=num_counting,
            target_qubits=[num_counting, num_counting + 1],
        )
        assert_equal(circuit4.num_qubits, num_counting + 2)

    def test_qpe_controlled_gate_powers(self):
        """
        Test if QPE correctly applies powers of U
        """
        phase = 0.25
        U = Qobj([[1, 0], [0, np.exp(1j * np.pi * phase)]])

        num_counting = 3
        circuit = qpe(
            U, num_counting_qubits=num_counting, target_qubits=num_counting
        )

        for i in range(num_counting):
            gate = circuit.instructions[num_counting + i].operation
            power = 2 ** (num_counting - i - 1)

            u_power = gate.target_gate.get_qobj()
            expected_u_power = U if power == 1 else U**power

            assert_((u_power - expected_u_power).norm() < 1e-12)

    def test_qpe_to_cnot_flag(self):
        """
        Test if to_cnot flag works correctly in QPE
        """
        U = sigmaz()
        num_counting = 2

        circuit1 = qpe(U, num_counting_qubits=num_counting, to_cnot=False)

        circuit2 = qpe(U, num_counting_qubits=num_counting, to_cnot=True)

        has_cnot = any(
            gate.operation.name == "CX" for gate in circuit2.instructions
        )
        assert_(has_cnot)
        assert_(len(circuit2.instructions) > len(circuit1.instructions))
