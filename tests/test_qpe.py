import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import unittest
from qutip import Qobj, sigmax, sigmay, sigmaz, identity, basis, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import gate_sequence_product, ControlledGate

from qutip_qip.algorithms.qpe import qpe, CustomGate, create_controlled_unitary

class TestQPE(unittest.TestCase):
    """
    A test class for the Quantum Phase Estimation implementation
    """
    
    def test_custom_gate(self):
        """
        Test if CustomGate correctly stores and returns the quantum object
        """
        # Create a simple unitary
        U = Qobj([[0, 1], [1, 0]])  # X gate
        
        # Create custom gate
        custom = CustomGate(targets=[0], U=U)
        
        # Check if get_compact_qobj returns the correct operator
        qobj = custom.get_compact_qobj()
        assert_((qobj - U).norm() < 1e-12)
        
        # Check targets are stored correctly
        assert_equal(custom.targets, [0])
        
    def test_controlled_unitary(self):
        """
        Test if create_controlled_unitary creates a correct controlled gate
        """
        # Create a simple unitary
        U = Qobj([[0, 1], [1, 0]])  # X gate
        
        # Create controlled unitary
        controlled_u = create_controlled_unitary(controls=[0], targets=[1], U=U)
        
        # Check properties
        assert_equal(controlled_u.controls, [0])
        assert_equal(controlled_u.targets, [1])
        assert_equal(controlled_u.control_value, 1)
        
        # Verify target_gate is CustomGate class
        assert_(controlled_u.target_gate == CustomGate)
        
        # Check kwargs contains U
        assert_('U' in controlled_u.kwargs)
        assert_((controlled_u.kwargs['U'] - U).norm() < 1e-12)
        
    def test_qpe_validation(self):
        """
        Test input validation in qpe function
        """
        U = sigmaz()
        
        # Test with invalid counting qubits
        with self.assertRaises(ValueError):
            qpe(U, num_counting_qubits=0)
            
        # Test with invalid unitary dimension
        invalid_U = Qobj([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3x3 matrix
        with self.assertRaises(ValueError):
            qpe(invalid_U, num_counting_qubits=3)
            
    def test_qpe_circuit_structure(self):
        """
        Test if qpe creates circuit with correct structure
        """
        # Simple unitary: Z gate with eigenvalues {+1, -1}
        U = sigmaz()
        
        # Create QPE circuit with 3 counting qubits
        num_counting = 3
        circuit = qpe(U, num_counting_qubits=num_counting, target_qubits=num_counting)
        
        # Total qubits should be num_counting + 1 (for target)
        assert_equal(circuit.N, num_counting + 1)
        
        # First num_counting gates should be Hadamard (SNOT)
        for i in range(num_counting):
            assert_equal(circuit.gates[i].targets, [i])
        
        # Next num_counting gates should be controlled unitaries
        for i in range(num_counting):
            gate = circuit.gates[num_counting + i]
            # Check it's a ControlledGate instance
            assert_(isinstance(gate, ControlledGate))
            # Control should be on counting qubit i
            assert_equal(gate.controls, [i])
            # Target should be on the target qubit
            assert_equal(gate.targets, [num_counting])
            
    def test_qpe_different_target_specifications(self):
        """
        Test QPE with different ways of specifying target qubits
        """
        U = sigmaz()
        num_counting = 2
        
        # Test with target as integer
        circuit1 = qpe(U, num_counting_qubits=num_counting, target_qubits=num_counting)
        assert_equal(circuit1.N, num_counting + 1)
        
        # Test with target as list
        circuit2 = qpe(U, num_counting_qubits=num_counting, target_qubits=[num_counting])
        assert_equal(circuit2.N, num_counting + 1)
        
        # Test with None (should auto-determine based on U)
        circuit3 = qpe(U, num_counting_qubits=num_counting, target_qubits=None)
        assert_equal(circuit3.N, num_counting + 1)
        
        # Test with multiple targets
        # Create a 4x4 unitary (2 qubits)
        U2 = tensor(sigmaz(), sigmaz())
        circuit4 = qpe(U2, num_counting_qubits=num_counting, target_qubits=[num_counting, num_counting+1])
        assert_equal(circuit4.N, num_counting + 2)
        
    def test_qpe_controlled_gate_powers(self):
        """
        Test if QPE correctly applies powers of U
        """
        # Use identity with phase for clearer testing
        phase = 0.25  # π/2 phase
        U = Qobj([[1, 0], [0, np.exp(1j * np.pi * phase)]])
        
        num_counting = 3
        circuit = qpe(U, num_counting_qubits=num_counting, target_qubits=num_counting)
        
        # Check the controlled unitary gates
        for i in range(num_counting):
            gate = circuit.gates[num_counting + i]
            # Calculate expected power
            power = 2**(num_counting - i - 1)
            
            # Extract U^power from the gate
            u_power = gate.kwargs['U']
            
            # Calculate expected U^power
            expected_u_power = U if power == 1 else U ** power
            
            # Check that they match
            assert_((u_power - expected_u_power).norm() < 1e-12)
            
    def test_qpe_to_cnot_flag(self):
        """
        Test if to_cnot flag works correctly in QPE
        """
        U = sigmaz()
        num_counting = 2
        
        # Test with to_cnot=False (default)
        circuit1 = qpe(U, num_counting_qubits=num_counting, to_cnot=False)
        
        # Test with to_cnot=True
        circuit2 = qpe(U, num_counting_qubits=num_counting, to_cnot=True)
        
        # Circuit with to_cnot=True should have CNOT gates
        has_cnot = any(gate.name == "CNOT" for gate in circuit2.gates)
        assert_(has_cnot)
        
        # Difference in gate count (circuit2 should have more gates due to decomposition)
        assert_(len(circuit2.gates) > len(circuit1.gates))