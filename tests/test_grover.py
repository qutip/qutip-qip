import numpy as np
import pytest
from qutip import basis, tensor, Qobj
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate
from qutip_qip.algorithms.grover import grover, grover_oracle


class TestGrover:
    def test_grover_oracle_unitary(self):
        """
        Test that grover_oracle produces the correct phase-flip unitary.
        """
        n_qubits = 3
        marked_states = [1, 5]  # |001> and |101>

        qc_oracle = grover_oracle(n_qubits, marked_states)

        U_sim = qc_oracle.compute_unitary()

        dims = [[2] * n_qubits, [2] * n_qubits]
        N = 2**n_qubits
        diag = np.ones(N)
        for s in marked_states:
            diag[s] = -1
        U_expected = Qobj(np.diag(diag), dims=dims)

        assert (U_sim - U_expected).norm() < 1e-6

    def test_grover_oracle_bounds_error(self):
        """Test that grover_oracle raises error for out-of-bounds states."""
        with pytest.raises(ValueError, match="out of bounds"):
            grover_oracle(2, [4])  # 2 qubits only go up to state 3

    def test_grover_2_qubit(self):
        """
        Full algorithm test: 2 qubits, searching for |11> (state 3).
        Optimal iterations = 1.
        """
        n_qubits = 2
        target_state = 3  # |11>

        oracle = grover_oracle(n_qubits, target_state)
        qc = grover(oracle, n_qubits, 1)

        U_grover = qc.compute_unitary()
        psi0 = tensor([basis(2, 0)] * n_qubits)  # Start at |00>
        psi_final = U_grover * psi0

        # Expected state: |11>
        psi_expected = basis(2**n_qubits, target_state)
        dims = [[2] * n_qubits, [1] * n_qubits]
        psi_expected.dims = dims

        fidelity = abs(psi_final.overlap(psi_expected)) ** 2
        assert fidelity > 0.999999

    def test_grover_3_qubit_multiple_targets(self):
        """
        Test 3 qubits with 2 marked states: |011> (3) and |101> (5).
        """
        n_qubits = 3
        marked = [3, 5]

        oracle = grover_oracle(n_qubits, marked)

        # For N=8, M=2, theta = 30 deg. One iteration rotates to 90 deg (solution).
        qc = grover(oracle, n_qubits, len(marked))

        U_grover = qc.compute_unitary()
        psi0 = tensor([basis(2, 0)] * n_qubits)
        psi_final = U_grover * psi0

        # Check probability of measuring EITHER 3 or 5
        prob_3 = abs(psi_final.overlap(basis(2**n_qubits, 3))) ** 2
        prob_5 = abs(psi_final.overlap(basis(2**n_qubits, 5))) ** 2

        total_success_prob = prob_3 + prob_5
        assert total_success_prob > 0.999999

    def test_grover_custom_qubit_indices(self):
        """
        Integration check: Run Grover on qubits [1, 2] of a 4-qubit system [0, 1, 2, 3].
        Qubits 0 and 3 should remain Identity.
        """

        sys_qubits = 4
        search_qubits = [1, 2]
        target_local_state = 3  # |11> on the 2 search qubits

        oracle = grover_oracle(search_qubits, target_local_state)

        # N=4, M=1. Optimal iterations = 1.
        qc = grover(oracle, search_qubits, 1)

        assert qc.num_qubits == 3  # max(1,2) + 1 = 3

        U_grover = qc.compute_unitary()

        # We simulate on 3 qubits (0, 1, 2). q0 is idle. q1,q2 are grover.
        psi0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))  # |000>
        psi_final = U_grover * psi0

        # Expected: |0> (idle) tensor |11> (grover result)
        psi_expected = tensor(basis(2, 0), basis(2, 1), basis(2, 1))

        fidelity = abs(psi_final.overlap(psi_expected)) ** 2
        assert fidelity > 0.9999

    def test_grover_oracle_types(self):
        """Test that the main function accepts Gate and Qobj oracles."""
        qubits = [0]

        # Qobj Oracle (Z gate for 1 qubit marks state |1>)
        oracle_qobj = Qobj([[1, 0], [0, -1]])
        qc_a = grover(oracle_qobj, qubits, 1)
        assert len(qc_a.gates) > 0

        # Gate Oracle
        oracle_gate = Gate("Z", targets=[0])
        qc_b = grover(oracle_gate, qubits, 1)
        assert len(qc_b.gates) > 0

        # Compare unitaries
        assert (qc_a.compute_unitary() - qc_b.compute_unitary()).norm() < 1e-6
