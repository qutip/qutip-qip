import pytest
import functools
import itertools
import numpy as np
import qutip
from qutip_qip.vqa import (
    VQA,
    VQABlock,
    ParameterizedHamiltonian,
    OptimizationResult,
)


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
        assert (
            block.get_unitary([angle]) == (-1j * angle * qutip.sigmax()).expm()
        )
        assert block.get_free_parameters_num() == 1
        block = VQABlock(qutip.sigmaz())
        assert (
            block.get_unitary([angle]) == (-1j * angle * qutip.sigmaz()).expm()
        )
        block = VQABlock(qutip.sigmay())
        assert (
            block.get_unitary([angle]) == (-1j * angle * qutip.sigmay()).expm()
        )

    @pytest.mark.parametrize("angle", [1, 2, 3])
    def test_unitary_function(self, angle):
        block = VQABlock(lambda t: (t * -1j * qutip.sigmax()).expm())
        assert (
            block.get_unitary([angle]) == (-1j * angle * qutip.sigmax()).expm()
        )


class TestVQA:
    """
    Test class for the VQA class
    """

    @pytest.mark.parametrize("n", [[[0, 1], [1.5, 1]], [[1, 0], [1, 1.5]]])
    def test_initialization_bad_input(self, n):
        """
        Test numerical inputs for initializing VQA
        """
        vqa = VQA(num_qubits=1, num_layers=1)
        with pytest.raises(ValueError):
            vqa = VQA(num_qubits=n[0][0], num_layers=n[0][1])
        with pytest.raises(TypeError):
            vqa = VQA(num_qubits=n[1][0], num_layers=n[1][1])

    @pytest.mark.parametrize("method", ["OBSERVABLE", "STATE", "BITSTRING"])
    def test_cost_methods(self, method):
        vqa = VQA(num_qubits=1, num_layers=1, cost_method=method)

    @pytest.mark.parametrize("method", ["Something else", 1, [0]])
    def test_invalid_cost_methods(self, method):
        with pytest.raises(ValueError):
            vqa = VQA(num_qubits=1, num_layers=1, cost_method=method)


class TestVQACircuit:
    """
    Test class for circuits made from the VQA and VQABlock combination
    """

    def test_initialize(self):
        block = VQABlock(qutip.sigmax(), is_unitary=True)
        vqa = VQA(num_qubits=1)
        final_state = vqa.get_final_state([])
        assert final_state == qutip.basis(2, 0)
        vqa.add_block(block)
        final_state = vqa.get_final_state([])
        assert final_state == qutip.basis(2, 1)

    def test_layer_repetitions(self):
        """
        tests layers are ordered and repeated correctly
        """
        vqa = VQA(num_qubits=1, num_layers=3)
        initial_block = VQABlock(qutip.sigmax(), initial=True)
        h_block = VQABlock("SNOT", targets=[0])
        vqa.add_block(initial_block)
        vqa.add_block(h_block)
        # Expect [X, H, H, H] for the three layers
        blocks = vqa.get_block_series()
        assert blocks[0] == initial_block
        for i in range(1, 4):
            assert blocks[i].name == h_block.name

    def test_parameterized_circuit(self):
        block = VQABlock(qutip.sigmax())
        vqa = VQA(num_qubits=1)
        final_state = vqa.get_final_state([0])
        assert final_state == qutip.basis(2, 0)
        vqa.add_block(block)
        final_state = vqa.get_final_state([np.pi / 2])
        assert final_state == qutip.Qobj([[0], [-1j]])

    def test_trivial_optimization(self):
        """
        tests trivial optimization case where the initial conditinos
        are already optimal for the problem
        """
        block = VQABlock(qutip.sigmax())
        # test the STATE type of cost function
        vqa = VQA(num_qubits=1, cost_method="STATE")
        vqa.add_block(block)
        # try to reach the |1> state from the |0> state
        vqa.cost_func = lambda s: 1 - s.overlap(qutip.basis(2, 1)).real
        res = vqa.optimize_parameters(initial=[np.pi / 2])
        assert res.get_top_bitstring() == "|1>"

    def test_layer_by_layer(self):
        """
        tests trivial optimization going layer-by-layer
        """
        vqa = VQA(num_qubits=1, cost_method="STATE", num_layers=4)
        block = VQABlock(qutip.sigmax())
        vqa.add_block(block)
        vqa.cost_func = lambda s: 1 - s.overlap(qutip.basis(2, 1)).real
        res = vqa.optimize_parameters(
            initial=[np.pi / 2, 0, 0, 0], layer_by_layer=True
        )
        assert res.get_top_bitstring() == "|1>"

    @pytest.mark.parametrize("use_jac", [True, False])
    def test_bfgs_optimization(self, use_jac):
        """
        test bfgs optimizer, starting from a close initial guess
        """
        block = VQABlock(qutip.sigmax())
        vqa = VQA(num_qubits=1, cost_method="OBSERVABLE")
        vqa.add_block(block)
        # try to reach the |1> state from the |0> state
        vqa.cost_observable = qutip.sigmaz()
        res = vqa.optimize_parameters(
            initial=[np.pi / 2 + 0.2], method="BFGS", use_jac=use_jac
        )
        assert res.get_top_bitstring() == "|1>"
        # check we actually found the function minimum
        assert round(res.res.x[0], 2) == round(np.pi / 2, 2)

    def test_parameterized_hamiltonian_blocks(self):
        """
        Test that parameterized hamiltonian blocks correctly
        apply parameters
        """
        # Hamiltonian that looks like (t_1*X  +  t_2*Z)
        block = VQABlock(
            ParameterizedHamiltonian([qutip.sigmax(), qutip.sigmaz()])
        )
        vqa = VQA(num_qubits=1, num_layers=2)
        vqa.add_block(block)
        # Do (pi/2*X + 0*Z) and then (0*X + pi/2*Z)
        final_state = vqa.get_final_state([np.pi / 2, 0, 0, np.pi / 2])
        # expect |1>
        assert final_state == qutip.basis(2, 1)

    def test_parameterized_hamiltonian_frechet_derivative(self):
        """
        Test gradient-based optimization on parameterized Hamiltonian blocks
        """
        vqa = VQA(num_qubits=1)
        vqa.cost_observable = qutip.sigmaz()
        block = VQABlock(ParameterizedHamiltonian([qutip.sigmax()]))
        vqa.add_block(block)
        res = vqa.optimize_parameters(
            initial=[np.pi / 2 + 0.2], method="BFGS", use_jac=True
        )
        assert res.get_top_bitstring() == "|1>"

    @pytest.mark.parametrize("todo", [False, True])
    def test_plot(self, todo):
        """
        Check plotting function returns without error
        """
        plt = pytest.importorskip("matplotlib.pyplot")
        # Only test on environments that have the matplotlib dependency
        vqa = VQA(num_qubits=4, num_layers=1, cost_method="STATE")
        for i in range(4):
            vqa.add_block(VQABlock("X", targets=[i]))
        vqa.cost_func = lambda s: 0
        res = vqa.optimize_parameters()
        res.plot(top_ten=todo, display=False)

    def test_bitstring_cost(self):
        "Check the bitstring sampling function"
        vqa = VQA(num_qubits=1, cost_method="BITSTRING")
        vqa.add_block(VQABlock(qutip.sigmax()))
        # target the |1> state by giving the "1" string a cost of 0
        vqa.cost_func = lambda s: 1 - int(s)
        res = vqa.optimize_parameters(initial=[np.pi / 2 + 1e-3])
        assert res.get_top_bitstring() == "|1>"

    def test_optimization_errors(self):
        """
        Tests for value errors relating to optimization procedure
        """
        vqa = VQA(num_qubits=1)
        vqa.add_block(VQABlock(qutip.sigmax()))
        vqa.cost_observable = qutip.sigmax()
        with pytest.raises(ValueError):
            # Invalid initialization string
            res = vqa.optimize_parameters(initial="something else")
        with pytest.raises(ValueError):
            # Incorrect number of parameters
            res = vqa.optimize_parameters(initial=[1, 1])
        # Valid initialization string
        res = vqa.optimize_parameters(initial="ones")
        assert res is not None


class TestOptimizationResult:
    """
    Test class for the OptimizationResult class
    """

    def test_label_to_sets(self):
        """
        Test partitioning a set of elements based on a bitstring.
        e.g. [1, 2, 3] with the bitstring |010> should become
        '{1, 3} {2}'.
        """
        # Fake scipy res object
        class fakeRes:
            pass

        fakeScipyRes = fakeRes()
        fakeRes.x = [1]
        fakeRes.fun = None
        fakeRes.nfev = None
        # Problem instance
        S = [1, 2, 3]
        result = OptimizationResult(fakeScipyRes, qutip.basis(8, 1))
        assert result._label_to_sets(S, "|010>") == "{1, 3} {2}"
