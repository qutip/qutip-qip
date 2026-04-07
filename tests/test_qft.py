from numpy.testing import assert_, assert_equal

from qutip_qip.algorithms.qft import qft, qft_steps, qft_gate_sequence
from qutip_qip.operations import gate_sequence_product
from qutip_qip.operations.gates import CPHASE, H, SWAP


class TestQFT:
    """
    A test class for the QuTiP functions for QFT
    """

    def testQFTComparison(self):
        """
        qft: compare qft and product of qft steps
        """
        for N in range(1, 5):
            U1 = qft(N)
            U2 = gate_sequence_product(qft_steps(N))
            assert_((U1 - U2).norm() < 1e-12)

    def testQFTGateSequenceNoSwapping(self):
        """
        qft: Inspect key properties of gate sequences of length N,
        with swapping disabled.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=False)
            assert_equal(circuit.num_qubits, N)

            totsize = N * (N + 1) / 2
            assert_equal(len(circuit.instructions), totsize)

            snots = sum(g.operation == H for g in circuit.instructions)
            assert_equal(snots, N)

            phases = sum(isinstance(g.operation, CPHASE) for g in circuit.instructions)
            assert_equal(phases, N * (N - 1) / 2)

    def testQFTGateSequenceWithSwapping(self):
        """
        qft: Inspect swap gates added to gate sequences if
        swapping is enabled.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=True)

            phases = int(N * (N + 1) / 2)
            swaps = int(N // 2)
            assert_equal(len(circuit.instructions), phases + swaps)

            for i in range(phases, phases + swaps):
                assert circuit.instructions[i].operation == SWAP

    def testQFTGateSequenceWithCNOT(self):
        """
        qft: Inspect swap gates added to gate sequences if
        it is decomposed into cnot.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=False, to_cnot=True)

        assert not any(
            [isinstance(ins.operation, CPHASE) for ins in circuit.instructions]
        )
