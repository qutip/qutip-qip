from copy import deepcopy
from packaging.version import parse as parse_version
import pytest
import functools
import itertools
import numpy as np
import qutip
from qutip_qip.operations import gates
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import (
    X, Y, Z, RX, RY, RZ, H, SQRTNOT, S, T, QASMU, CNOT, CPHASE, ISWAP, SWAP,
    CZ, SQRTSWAP, SQRTISWAP, SWAPALPHA, SWAPALPHA, MS, TOFFOLI, FREDKIN,
    BERKELEY, R, RZX, expand_operator)


def _permutation_id(permutation):
    return str(len(permutation)) + "-" + "".join(map(str, permutation))


def _infidelity(a, b):
    """Infidelity between two kets."""
    return 1 - abs(a.overlap(b))


def _make_random_three_qubit_gate():
    """Create a random three-qubit gate."""
    if parse_version(qutip.__version__) < parse_version('5.dev'):
        operation = qutip.rand_unitary(8, dims=[[2] * 3] * 2)
    else:
        operation = qutip.rand_unitary([2] * 3)

    def gate(N=None, controls=None, target=None):
        if N is None:
            return operation
        return expand_operator(operation, dims=[2]*N, targets=controls + [target])
    return gate


def _tensor_with_entanglement(all_qubits, entangled, entangled_locations):
    """
    Create a full tensor product when a subspace component is already in an
    entangled state.  The locations in `all_qubits` which are the entangled
    points in the output are ignored and can take any value.

    For example,
        _tensor_with_entanglement([|a>, |b>, |c>, |d>], (|00> + |11>), [0, 2])
    should product a tensor product like (|0b0d> + |1b1d>), i.e. qubits 0 and 2
    in the final output are entangled, but the others are still separable.

    Parameters:
        all_qubits: list of kets --
            A list of separable parts to tensor together.  States that are in
            the locations referred to by `entangled_locations` are completely
            ignored.
        entangled: tensor-product ket -- the full entangled subspace
        entangled_locations: list of int --
            The locations that the qubits in the entangled subspace should be
            in in the final tensor-product space.
    """
    n_entangled = len(entangled.dims[0])
    n_separable = len(all_qubits) - n_entangled
    separable = all_qubits.copy()
    # Remove in reverse order so subsequent deletion locations don't change.
    for location in sorted(entangled_locations, reverse=True):
        del separable[location]
    # Can't separate out entangled states to pass to tensor in the right places
    # immediately, so tensor in at one end and then permute into place.
    out = qutip.tensor(*separable, entangled)
    permutation = list(range(n_separable))
    current_locations = range(n_separable, n_separable + n_entangled)
    # Sort to prevert later insertions changing previous locations.
    insertions = sorted(zip(entangled_locations, current_locations),
                        key=lambda x: x[0])
    for out_location, current_location in insertions:
        permutation.insert(out_location, current_location)
    return out.permute(permutation)


def _apply_permutation(permutation):
    """
    Permute the given permutation into the order denoted by its elements, i.e.
        out[0] = permutation[permutation[0]]
        out[1] = permutation[permutation[1]]
        ...

    This function is its own inverse.
    """
    out = [None] * len(permutation)
    for value, location in enumerate(permutation):
        out[location] = value
    return out


class TestExplicitForm:
    def test_swap(self):
        states = [qutip.rand_ket(2) for _ in [None]*2]
        start = qutip.tensor(states)
        swapped = qutip.tensor(states[::-1])
        swap = gates.swap()
        assert _infidelity(swapped, swap*start) < 1e-12
        assert _infidelity(start, swap*swap*start) < 1e-12

    @pytest.mark.parametrize('permutation',
                             tuple(itertools.permutations([0, 1, 2])),
                             ids=_permutation_id)
    def test_toffoli(self, permutation):
        test = expand_operator(
            gates.toffoli(), dims=[2] * 3, targets=permutation)
        base = (qutip.tensor(1 - qutip.basis([2, 2], [1, 1]).proj(),
                             qutip.qeye(2))
                + qutip.tensor(qutip.basis([2, 2], [1, 1]).proj(),
                               qutip.sigmax()))
        # Iterate the permutation once, equivalent to finding its inverse,
        # because for us, `permutation[n]` is where qubit `n` should be placed,
        # whereas `Qobj.permute` interprets that qubit `n` should be at the
        # location `i` where `permutation[i] == n`.
        expected = base.permute(_apply_permutation(permutation))
        assert test == expected

    @pytest.mark.parametrize(['angle', 'expected'], [
        pytest.param(np.pi, -1j*qutip.tensor(qutip.sigmax(), qutip.sigmax()),
                     id="pi"),
        pytest.param(2*np.pi, -qutip.qeye([2, 2]), id="2pi"),
    ])
    def test_molmer_sorensen(self, angle, expected):
        np.testing.assert_allclose(gates.molmer_sorensen(angle).full(),
                                   expected.full(), atol=1e-15)

    @pytest.mark.parametrize(["gate", "n_angles"], [
        pytest.param(gates.rx, 1, id="Rx"),
        pytest.param(gates.ry, 1, id="Ry"),
        pytest.param(gates.rz, 1, id="Rz"),
        pytest.param(gates.phasegate, 1, id="phase"),
        pytest.param(gates.qrot, 2, id="Rabi rotation"),
    ])
    def test_zero_rotations_are_identity(self, gate, n_angles):
        np.testing.assert_allclose(
            np.eye(2), gate(*([0]*n_angles)).full(), atol=1e-15)


class TestCliffordGroup:
    """
    Test a sufficient set of conditions to prove that we have a full Clifford
    group for a single qubit.
    """
    clifford = list(gates.qubit_clifford_group())
    pauli = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    def test_single_qubit_group_dimension_is_24(self):
        assert len(self.clifford) == 24

    def test_all_elements_different(self):
        clifford = [gate for gate in self.clifford]
        for i, gate in enumerate(clifford):
            for other in clifford[i+1:]:
                # Big tolerance because we actually want to test the inverse.
                fid = qutip.average_gate_fidelity(gate, other)
                assert not np.allclose(fid, 1., atol=1e-3)

    @pytest.mark.parametrize("gate", gates.qubit_clifford_group())
    def test_gate_normalises_pauli_group(self, gate):
        """
        Test the fundamental definition of the Clifford group, i.e. that it
        normalises the Pauli group.
        """
        # Assert that each Clifford gate maps the set of Pauli gates back onto
        # itself (though not necessarily in order).  This condition is no
        # stronger than simply considering each (gate, Pauli) pair separately.
        pauli_gates = deepcopy(self.pauli)
        normalised = [gate * pauli * gate.dag() for pauli in self.pauli]
        for gate in normalised:
            for i, pauli in enumerate(pauli_gates):
                # if np.allclose(gate.full(), pauli.full(), atol=1e-10):
                if np.allclose(qutip.average_gate_fidelity(gate, pauli), 1):
                    del pauli_gates[i]
                    break
        assert len(pauli_gates) == 0


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestGateExpansion:
    """
    Test that gates act correctly when supplied with controls and targets, i.e.
    that they pick out the correct qubits to act on, the action is correct, and
    all other qubits are untouched.
    """
    n_qubits = 5

    @pytest.mark.parametrize(["gate", "n_angles"], [
        pytest.param(gates.rx, 1, id="Rx"),
        pytest.param(gates.ry, 1, id="Ry"),
        pytest.param(gates.rz, 1, id="Rz"),
        pytest.param(gates.x_gate, 0, id="X"),
        pytest.param(gates.y_gate, 0, id="Y"),
        pytest.param(gates.z_gate, 0, id="Z"),
        pytest.param(gates.s_gate, 0, id="S"),
        pytest.param(gates.t_gate, 0, id="T"),
        pytest.param(gates.phasegate, 1, id="phase"),
        pytest.param(gates.qrot, 2, id="Rabi rotation"),
    ])
    def test_single_qubit_rotation(self, gate, n_angles):
        base = qutip.rand_ket(2)
        angles = np.random.rand(n_angles) * 2*np.pi
        applied = gate(*angles) * base
        random = [qutip.rand_ket(2) for _ in [None]*(self.n_qubits - 1)]
        for target in range(self.n_qubits):
            start = qutip.tensor(random[:target] + [base] + random[target:])
            test = gate(*angles, self.n_qubits, target) * start
            expected = qutip.tensor(random[:target] + [applied]
                                    + random[target:])
            assert _infidelity(test, expected) < 1e-12

    @pytest.mark.parametrize(['gate', 'n_controls'], [
        pytest.param(gates.cnot, 1, id="cnot"),
        pytest.param(gates.cy_gate, 1, id="cY"),
        pytest.param(gates.cz_gate, 1, id="cZ"),
        pytest.param(gates.cs_gate, 1, id="cS"),
        pytest.param(gates.ct_gate, 1, id="cT"),
        pytest.param(gates.swap, 0, id="swap"),
        pytest.param(gates.iswap, 0, id="iswap"),
        pytest.param(gates.sqrtswap, 0, id="sqrt(swap)"),
        pytest.param(functools.partial(gates.molmer_sorensen, 0.5*np.pi, 0.), 0,
                     id="Molmer-Sorensen")
    ])
    def test_two_qubit(self, gate, n_controls):
        targets = [qutip.rand_ket(2) for _ in [None]*2]
        others = [qutip.rand_ket(2) for _ in [None]*self.n_qubits]
        reference = gate() * qutip.tensor(*targets)
        for q1, q2 in itertools.permutations(range(self.n_qubits), 2):
            qubits = others.copy()
            qubits[q1], qubits[q2] = targets
            args = [[q1, q2]] if n_controls == 0 else [q1, q2]
            test = gate(self.n_qubits, *args) * qutip.tensor(*qubits)
            expected = _tensor_with_entanglement(qubits, reference, [q1, q2])
            assert _infidelity(test, expected) < 1e-12

    @pytest.mark.parametrize(['gate', 'n_controls'], [
        pytest.param(gates.fredkin, 1, id="Fredkin"),
        pytest.param(gates.toffoli, 2, id="Toffoli"),
        pytest.param(_make_random_three_qubit_gate(), 2, id="random"),
    ])
    def test_three_qubit(self, gate, n_controls):
        targets = [qutip.rand_ket(2) for _ in [None]*3]
        others = [qutip.rand_ket(2) for _ in [None]*self.n_qubits]
        reference = gate() * qutip.tensor(targets)
        for q1, q2, q3 in itertools.permutations(range(self.n_qubits), 3):
            qubits = others.copy()
            qubits[q1], qubits[q2], qubits[q3] = targets
            args = [q1, [q2, q3]] if n_controls == 1 else [[q1, q2], q3]
            test = gate(self.n_qubits, *args) * qutip.tensor(*qubits)
            expected = _tensor_with_entanglement(qubits, reference,
                                                 [q1, q2, q3])
            assert _infidelity(test, expected) < 1e-12


class Test_expand_operator:
    # Conceptually, a lot of these tests are complete duplicates of
    # `TestGateExpansion`, except that they explicitly target an underlying
    # function in `qutip_qip.operations.gates` which (as of 2020-03-01) is not
    # called in the majority of cases---it appears to be newer than the
    # surrounding code, but the surrounding code wasn't updated.
    @pytest.mark.parametrize(
        'permutation',
        tuple(itertools.chain(*[
            itertools.permutations(range(k)) for k in [2, 3, 4]
        ])),
        ids=_permutation_id)
    def test_permutation_without_expansion(self, permutation):
        base = qutip.tensor([qutip.rand_unitary(2) for _ in permutation])
        test = gates.expand_operator(
            base,
            dims=[2] * len(permutation),
            targets=permutation)
        expected = base.permute(_apply_permutation(permutation))
        np.testing.assert_allclose(test.full(), expected.full(), atol=1e-15)

    @pytest.mark.parametrize('n_targets', range(1, 5))
    def test_general_qubit_expansion(self, n_targets):
        # Test all permutations with the given number of targets.
        n_qubits = 5
        if parse_version(qutip.__version__) < parse_version('5.dev'):
            operation = qutip.rand_unitary(2**n_targets, dims=[[2]*n_targets]*2)
        else:
            operation = qutip.rand_unitary([2]*n_targets)
        for targets in itertools.permutations(range(n_qubits), n_targets):
            expected = _tensor_with_entanglement([qutip.qeye(2)] * n_qubits,
                                                 operation, targets)
            test = gates.expand_operator(
                operation, dims=[2] * n_qubits, targets=targets)
            np.testing.assert_allclose(test.full(), expected.full(),
                                       atol=1e-15)

    def test_cnot_explicit(self):
        test = gates.expand_operator(
            gates.cnot(), dims=[2] * 3, targets=[2, 0]).full()
        expected = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0]])
        np.testing.assert_allclose(test, expected, atol=1e-15)

    def test_hadamard_explicit(self):
        test = gates.hadamard_transform(3).full()
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
            ]
        )
        expected = expected / np.sqrt(8)
        np.testing.assert_allclose(test, expected)

    @pytest.mark.parametrize('dimensions', [
        pytest.param([3, 4, 5], id="standard"),
        pytest.param([3, 3, 4, 4, 2], id="standard"),
        pytest.param([1, 2, 3], id="1D space"),
    ])
    def test_non_qubit_systems(self, dimensions):
        n_qubits = len(dimensions)
        for targets in itertools.permutations(range(n_qubits), 2):
            operators = [qutip.rand_unitary(dimension) if n in targets
                         else qutip.qeye(dimension)
                         for n, dimension in enumerate(dimensions)]
            expected = qutip.tensor(*operators)
            base_test = qutip.tensor(*[operators[x] for x in targets])
            test = gates.expand_operator(base_test, dims=dimensions,
                                         targets=targets)
            assert test.dims == expected.dims
            np.testing.assert_allclose(test.full(), expected.full())

    @pytest.mark.skipif(
        not parse_version(qutip.__version__) >= parse_version('5.dev'),
        reason=
        "Data type only exist in v5."
        )
    def test_dtype(self):
        expanded_qobj = expand_operator(gates.cnot(), dims=[2, 2, 2]).data
        assert isinstance(expanded_qobj, qutip.data.CSR)
        expanded_qobj = expand_operator(
            gates.cnot(), dims=[2, 2, 2], dtype="dense").data
        assert isinstance(expanded_qobj, qutip.data.Dense)


def test_gates_class():
    if parse_version(qutip.__version__) < parse_version('5.dev'):
        init_state = qutip.rand_ket(8, dims=[[2, 2, 2], [1, 1, 1]])
    else:
        init_state = qutip.rand_ket([2, 2, 2])

    circuit1 = QubitCircuit(3)
    circuit1.add_gate("X", 1)
    circuit1.add_gate("Y", 1)
    circuit1.add_gate("Z", 2)
    circuit1.add_gate("RX", 0, arg_value=np.pi/4)
    circuit1.add_gate("RY", 0, arg_value=np.pi/4)
    circuit1.add_gate("RZ", 1, arg_value=np.pi/4)
    circuit1.add_gate("H", 0)
    circuit1.add_gate("SQRTNOT", 0)
    circuit1.add_gate("S", 2)
    circuit1.add_gate("T", 1)
    circuit1.add_gate("R", 1, arg_value=(np.pi/4, np.pi/6))
    circuit1.add_gate("QASMU", 0, arg_value=(np.pi/4, np.pi/4, np.pi/4))
    circuit1.add_gate("CNOT", controls=0, targets=1)
    circuit1.add_gate("CPHASE", controls=0, targets=1, arg_value=np.pi/4)
    circuit1.add_gate("SWAP", [0, 1])
    circuit1.add_gate("ISWAP", [2, 1])
    circuit1.add_gate("CZ", controls=0, targets=2)
    circuit1.add_gate("SQRTSWAP", [2, 0])
    circuit1.add_gate("SQRTISWAP", [0, 1])
    circuit1.add_gate("SWAPALPHA", [1, 2], arg_value=np.pi/4)
    circuit1.add_gate("MS", [1, 0], arg_value=(np.pi/4, np.pi/7))
    circuit1.add_gate("TOFFOLI", [2, 0, 1])
    circuit1.add_gate("FREDKIN", [0, 1, 2])
    circuit1.add_gate("BERKELEY", [1, 0])
    circuit1.add_gate("RZX", [1, 0], arg_value=1.)
    result1 = circuit1.run(init_state)

    circuit2 = QubitCircuit(3)
    circuit2.add_gate(X(1))
    circuit2.add_gate(Y(1))
    circuit2.add_gate(Z(2))
    circuit2.add_gate(RX(0, np.pi/4))
    circuit2.add_gate(RY(0, np.pi/4))
    circuit2.add_gate(RZ(1, np.pi/4))
    circuit2.add_gate(H(0))
    circuit2.add_gate(SQRTNOT(0))
    circuit2.add_gate(S(2))
    circuit2.add_gate(T(1))
    circuit2.add_gate(R(1, (np.pi/4, np.pi/6)))
    circuit2.add_gate(QASMU(0, (np.pi/4, np.pi/4, np.pi/4)))
    circuit2.add_gate(CNOT(0, 1))
    circuit2.add_gate(CPHASE(0, 1, np.pi/4))
    circuit2.add_gate(SWAP([0, 1]))
    circuit2.add_gate(ISWAP([2, 1]))
    circuit2.add_gate(CZ(0, 2))
    circuit2.add_gate(SQRTSWAP([2, 0]))
    circuit2.add_gate(SQRTISWAP([0, 1]))
    circuit2.add_gate(SWAPALPHA([1, 2], np.pi/4))
    circuit2.add_gate(MS([1, 0], (np.pi/4, np.pi/7)))
    circuit2.add_gate(TOFFOLI([2, 0, 1]))
    circuit2.add_gate(FREDKIN([0, 1, 2]))
    circuit2.add_gate(BERKELEY([1, 0]))
    circuit2.add_gate(RZX([1, 0], 1.))
    result2 = circuit2.run(init_state)

    assert pytest.approx(qutip.fidelity(result1, result2), 1.0e-6) == 1
