import pytest
import functools
import itertools
import numpy as np
import qutip
from qutip_qip.vqa import VQA, VQABlock, ParameterizedHamiltonian


class TestVQABlock:
    """
    Test class for the VQABlock class
    """
    def test_initial(self):
        block = VQABlock(qutip.sigmax())
        assert block.get_unitary([1]) == (-1j * qutip.sigmax()).expm()

    @pytest.mark.parametrize("angle", [1, 2, 3])
    def test_parameterization(self, angle):
        block = VQABlock(qutip.sigmax())
        assert block.get_unitary([angle]) \
               == (-1j * angle * qutip.sigmax()).expm()
        assert block.get_free_parameters() == 1
        block = VQABlock(qutip.sigmaz())
        assert block.get_unitary([angle]) \
               == (-1j * angle * qutip.sigmaz()).expm()
        block = VQABlock(qutip.sigmay())
        assert block.get_unitary([angle]) == \
               (-1j * angle * qutip.sigmay()).expm()

    @pytest.mark.parametrize("angle", [1, 2, 3])
    def test_unitary_function(self, angle):
        block = VQABlock(lambda t: (t*-1j*qutip.sigmax()).expm())
        assert block.get_unitary([angle]) == \
               (-1j * angle * qutip.sigmax()).expm()


class TestVQA:
    """
    Test class for the VQA class
    """
    @pytest.mark.parametrize("n", [[[0, 1], [1.5, 1]], [[1, 0], [1, 1.5]]])
    def test_initialization_bad_input(self, n):
        """
        Test numerical inputs for initializing VQA
        """
        vqa = VQA(n_qubits=1, n_layers=1)
        with pytest.raises(ValueError):
            vqa = VQA(n_qubits=n[0][0], n_layers=n[0][1])
        with pytest.raises(TypeError):
            vqa = VQA(n_qubits=n[1][0], n_layers=n[1][1])

    @pytest.mark.parametrize("method", ["OBSERVABLE", "STATE", "BITSTRING"])
    def test_cost_methods(self, method):
        vqa = VQA(n_qubits=1, n_layers=1, cost_method=method)

    @pytest.mark.parametrize("method", ["Something else", 1, [0]])
    def test_invalid_cost_methods(self, method):
        with pytest.raises(ValueError):
            vqa = VQA(n_qubits=1, n_layers=1, cost_method=method)


class TestVQACircuit:
    """
    Test class for circuits made from the VQA and VQABlock combination
    """
    def test_initialize(self):
        block = VQABlock(qutip.sigmax(), is_unitary=True)
        vqa = VQA(n_qubits=1)
        final_state = vqa.get_final_state([])
        assert final_state == qutip.basis(2, 0)
        vqa.add_block(block)
        final_state = vqa.get_final_state([])
        assert final_state == qutip.basis(2, 1)

    def test_parameterized_circuit(self):
        block = VQABlock(qutip.sigmax())
        vqa = VQA(n_qubits=1)
        final_state = vqa.get_final_state([0])
        assert final_state == qutip.basis(2, 0)
        vqa.add_block(block)
        final_state = vqa.get_final_state([np.pi/2])
        assert final_state == qutip.Qobj([[0], [-1j]])

    def test_trivial_optimization(self):
        """
        tests trivial optimization case where the initial conditinos
        are already optimal for the problem
        """
        block = VQABlock(qutip.sigmax())
        # test the STATE type of cost function
        vqa = VQA(n_qubits=1, cost_method="STATE")
        vqa.add_block(block)
        # try to reach the |1> state from the |0> state
        vqa.cost_func = lambda s: 1 - s.overlap(qutip.basis(2, 1)).real
        res = vqa.optimize_parameters(initial=[np.pi/2])
        assert res.get_top_bitstring() == "|1>"

    def test_bfgs_optimization(self):
        """
        test bfgs optimizer, starting from a close initial guess
        """
        block = VQABlock(qutip.sigmax())
        # test the STATE type of cost function
        vqa = VQA(n_qubits=1, cost_method="OBSERVABLE")
        vqa.add_block(block)
        # try to reach the |1> state from the |0> state
        vqa.cost_observable = qutip.sigmaz()
        res = vqa.optimize_parameters(
                initial=[np.pi/2 + 0.2],
                method="BFGS"
                )
        assert res.get_top_bitstring() == "|1>"
        # check we actually found the function minimum
        assert round(res.res.x[0], 2) == round(np.pi/2, 2)

    def test_parameterized_hamiltonian_blocks(self):
        # Hamiltonian that looks like (t_1*X  +  t_2*Z)
        block = VQABlock(
                ParameterizedHamiltonian([qutip.sigmax(), qutip.sigmaz()])
                )
        vqa = VQA(n_qubits=1, n_layers=2)
        vqa.add_block(block)
        # Do (pi/2*X + 0*Z) and then (0*X + pi/2*Z)
        final_state = vqa.get_final_state([np.pi/2, 0, 0, np.pi/2])
        # expect |1>
        assert final_state == qutip.basis(2, 1)
