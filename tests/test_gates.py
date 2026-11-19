from copy import deepcopy
import pytest
import itertools
import pickle

import numpy as np
import qutip

from qutip_qip.operations import (
    AngleUnitary,
    ControlledUnitary,
    Mz,
    Unitary,
    controlled,
    expand_operator,
    get_unitary_gate,
    hadamard_transform,
    qubit_clifford_group,
)
import qutip_qip.operations.gates as gates
from qutip_qip.operations.unitary import Inverse

rng = np.random.default_rng(seed=101)


def _permutation_id(permutation):
    return str(len(permutation)) + "-" + "".join(map(str, permutation))


def _infidelity(a, b):
    """Infidelity between two kets."""
    return 1 - abs(a.overlap(b))


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
    # Sort to prevent later insertions changing previous locations.
    insertions = sorted(zip(entangled_locations, current_locations), key=lambda x: x[0])
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
        states = [qutip.rand_ket(2) for _ in [None] * 2]
        start = qutip.tensor(states)
        swapped = qutip.tensor(states[::-1])
        swap = gates.SWAP.get_qobj()
        assert _infidelity(swapped, swap * start) < 1e-12
        assert _infidelity(start, swap * swap * start) < 1e-12

    @pytest.mark.parametrize(
        "permutation",
        tuple(itertools.permutations([0, 1, 2])),
        ids=_permutation_id,
    )
    def test_toffoli(self, permutation):
        test = expand_operator(
            gates.TOFFOLI.get_qobj(), dims=[2] * 3, targets=permutation
        )
        base = qutip.tensor(
            1 - qutip.basis([2, 2], [1, 1]).proj(), qutip.qeye(2)
        ) + qutip.tensor(qutip.basis([2, 2], [1, 1]).proj(), qutip.sigmax())
        # Iterate the permutation once, equivalent to finding its inverse,
        # because for us, `permutation[n]` is where qubit `n` should be placed,
        # whereas `Qobj.permute` interprets that qubit `n` should be at the
        # location `i` where `permutation[i] == n`.
        expected = base.permute(_apply_permutation(permutation))
        assert test == expected

    @pytest.mark.parametrize(
        ["angle", "expected"],
        [
            pytest.param(
                [np.pi, 0],
                -1j * qutip.tensor(qutip.sigmax(), qutip.sigmax()),
                id="pi",
            ),
            pytest.param([2 * np.pi, 0], -qutip.qeye([2, 2]), id="2pi"),
        ],
    )
    def test_molmer_sorensen(self, angle, expected):
        np.testing.assert_allclose(
            gates.MS(*angle).get_qobj().full(), expected.full(), atol=1e-15
        )

    @pytest.mark.parametrize(
        ["gate", "n_angles"],
        [
            pytest.param(gates.RX, 1, id="Rx"),
            pytest.param(gates.RY, 1, id="Ry"),
            pytest.param(gates.RZ, 1, id="Rz"),
            pytest.param(gates.PHASE, 1, id="phase"),
            pytest.param(gates.R, 2, id="Rabi rotation"),
        ],
    )
    def test_zero_rotations_are_identity(self, gate, n_angles):
        np.testing.assert_allclose(
            np.eye(2), gate(*[0] * n_angles).get_qobj().full(), atol=1e-15
        )


class TestCliffordGroup:
    """
    Test a sufficient set of conditions to prove that we have a full Clifford
    group for a single qubit.
    """

    with pytest.warns(DeprecationWarning):
        clifford = list(qubit_clifford_group())
    pauli = [qutip.qeye(2), qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    def test_single_qubit_group_dimension_is_24(self):
        assert len(self.clifford) == 24

    def test_all_elements_different(self):
        clifford = [gate for gate in self.clifford]
        for i, gate in enumerate(clifford):
            for other in clifford[i + 1 :]:
                # Big tolerance because we actually want to test the inverse.
                fid = qutip.average_gate_fidelity(gate, other)
                assert not np.allclose(fid, 1.0, atol=1e-3)

    @pytest.mark.parametrize("gate", clifford)
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestGateExpansion:
    """
    Test that gates act correctly when supplied with controls and targets, i.e.
    that they pick out the correct qubits to act on, the action is correct, and
    all other qubits are untouched.
    """

    n_qubits = 5

    @pytest.mark.parametrize(
        ["gate", "n_angles"],
        [
            pytest.param(gates.RX, 1, id="Rx"),
            pytest.param(gates.RY, 1, id="Ry"),
            pytest.param(gates.RZ, 1, id="Rz"),
            pytest.param(gates.X, 0, id="X"),
            pytest.param(gates.Y, 0, id="Y"),
            pytest.param(gates.Z, 0, id="Z"),
            pytest.param(gates.S, 0, id="S"),
            pytest.param(gates.T, 0, id="T"),
            pytest.param(gates.PHASE, 1, id="phase"),
            pytest.param(gates.R, 2, id="Rabi rotation"),
        ],
    )
    def test_single_qubit_rotation(self, gate: type[Unitary], n_angles: int):
        base = qutip.rand_ket(2)
        if n_angles > 0:
            angles = 2 * np.pi * (np.random.rand(n_angles))
            gate = gate(*angles)

        applied = gate.get_qobj() * base
        random = [qutip.rand_ket(2) for _ in [None] * (self.n_qubits - 1)]

        for target in range(self.n_qubits):
            start = qutip.tensor(random[:target] + [base] + random[target:])
            test = (
                expand_operator(
                    gate.get_qobj(), targets=target, dims=[2] * self.n_qubits
                )
                * start
            )
            expected = qutip.tensor(random[:target] + [applied] + random[target:])
            assert _infidelity(test, expected) < 1e-12

    @pytest.mark.parametrize(
        ["gate", "n_controls"],
        [
            pytest.param(gates.CX, 1, id="CX"),
            pytest.param(gates.CY, 1, id="CY"),
            pytest.param(gates.CZ, 1, id="CZ"),
            pytest.param(gates.CS, 1, id="CS"),
            pytest.param(gates.CT, 1, id="CT"),
            pytest.param(gates.SWAP, 0, id="SWAP"),
            pytest.param(gates.ISWAP, 0, id="ISWAP"),
            pytest.param(gates.SQRTSWAP, 0, id="SQRTSWAP"),
            pytest.param(gates.MS(0.5 * np.pi, 0.0), 0, id="Molmer-Sorensen"),
        ],
    )
    def test_two_qubit(self, gate, n_controls):
        targets = [qutip.rand_ket(2) for _ in [None] * 2]
        others = [qutip.rand_ket(2) for _ in [None] * self.n_qubits]
        reference = gate.get_qobj() * qutip.tensor(*targets)

        for q1, q2 in itertools.permutations(range(self.n_qubits), 2):
            qubits = others.copy()
            qubits[q1], qubits[q2] = targets
            test = expand_operator(
                gate.get_qobj(), dims=[2] * self.n_qubits, targets=[q1, q2]
            ) * qutip.tensor(*qubits)
            expected = _tensor_with_entanglement(qubits, reference, [q1, q2])
            assert _infidelity(test, expected) < 1e-12

    random_gate = get_unitary_gate("random", qutip.rand_unitary([2] * 1))
    RandomThreeQubitGate = controlled(random_gate, 2)

    @pytest.mark.parametrize(
        ["gate", "n_controls"],
        [
            pytest.param(gates.FREDKIN, 1, id="Fredkin"),
            pytest.param(gates.TOFFOLI, 2, id="Toffoli"),
            pytest.param(RandomThreeQubitGate, 2, id="random"),
        ],
    )
    def test_three_qubit(self, gate: type[Unitary], n_controls):
        targets = [qutip.rand_ket(2) for _ in [None] * 3]
        others = [qutip.rand_ket(2) for _ in [None] * self.n_qubits]
        reference = gate.get_qobj() * qutip.tensor(targets)

        for q1, q2, q3 in itertools.permutations(range(self.n_qubits), 3):
            qubits = others.copy()
            qubits[q1], qubits[q2], qubits[q3] = targets
            test = expand_operator(
                gate.get_qobj(), targets=[q1, q2, q3], dims=[2] * self.n_qubits
            ) * qutip.tensor(*qubits)
            expected = _tensor_with_entanglement(qubits, reference, [q1, q2, q3])
            assert _infidelity(test, expected) < 1e-12


class Test_expand_operator:
    # Conceptually, a lot of these tests are complete duplicates of
    # `TestGateExpansion`, except that they explicitly target an underlying
    # function in `qutip_qip.operations.gates` which (as of 2020-03-01) is not
    # called in the majority of cases---it appears to be newer than the
    # surrounding code, but the surrounding code wasn't updated.
    @pytest.mark.parametrize(
        "permutation",
        tuple(itertools.chain(*[itertools.permutations(range(k)) for k in [2, 3, 4]])),
        ids=_permutation_id,
    )
    def test_permutation_without_expansion(self, permutation):
        base = qutip.tensor([qutip.rand_unitary(2) for _ in permutation])
        test = expand_operator(base, dims=[2] * len(permutation), targets=permutation)
        expected = base.permute(_apply_permutation(permutation))
        np.testing.assert_allclose(test.full(), expected.full(), atol=1e-15)

    @pytest.mark.parametrize("n_targets", range(1, 5))
    def test_general_qubit_expansion(self, n_targets):
        # Test all permutations with the given number of targets.
        n_qubits = 5
        operation = qutip.rand_unitary([2] * n_targets)

        for targets in itertools.permutations(range(n_qubits), n_targets):
            expected = _tensor_with_entanglement(
                [qutip.qeye(2)] * n_qubits, operation, targets
            )
            test = expand_operator(operation, dims=[2] * n_qubits, targets=targets)
            np.testing.assert_allclose(test.full(), expected.full(), atol=1e-15)

    def test_cnot_explicit(self):
        test = expand_operator(gates.CX.get_qobj(), dims=[2] * 3, targets=[2, 0]).full()
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        np.testing.assert_allclose(test, expected, atol=1e-15)

    def test_hadamard_explicit(self):
        test = hadamard_transform(3).full()
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

    @pytest.mark.parametrize(
        "dimensions",
        [
            pytest.param([3, 4, 5], id="standard"),
            pytest.param([3, 3, 4, 4, 2], id="standard"),
            pytest.param([1, 2, 3], id="1D space"),
        ],
    )
    def test_non_qubit_systems(self, dimensions):
        n_qubits = len(dimensions)
        for targets in itertools.permutations(range(n_qubits), 2):
            operators = [
                (
                    qutip.rand_unitary(dimension)
                    if n in targets
                    else qutip.qeye(dimension)
                )
                for n, dimension in enumerate(dimensions)
            ]
            expected = qutip.tensor(*operators)
            base_test = qutip.tensor(*[operators[x] for x in targets])
            test = expand_operator(base_test, dims=dimensions, targets=targets)
            assert test.dims == expected.dims
            np.testing.assert_allclose(test.full(), expected.full())

    def test_dtype(self):
        expanded_qobj = expand_operator(gates.CX.get_qobj(), dims=[2, 2, 2]).data
        assert isinstance(expanded_qobj, qutip.data.CSR)
        expanded_qobj = expand_operator(
            gates.CX.get_qobj(), dims=[2, 2, 2], dtype="dense"
        ).data
        assert isinstance(expanded_qobj, qutip.data.Dense)


rand_U = qutip.rand_unitary(dimensions=[2], seed=rng)


class U1(Unitary):
    num_qreg = 1

    @staticmethod
    def compute_qobj(dtype: str = "dense"):
        return rand_U.to(dtype)


class U2(AngleUnitary):
    num_qreg = 1
    theta: float

    def compute_qobj(self, dtype: str = "dense"):
        return qutip.Qobj(
            [
                [np.exp(-1j * self.theta), 0],
                [0, np.exp(1j * self.theta)],
            ],
            dtype=dtype,
        )

    def compute_inverse(self):
        return U2(-self.theta)


GATES = [
    gates.X,
    gates.Y,
    gates.Z,
    gates.H,
    gates.S,
    gates.Sdag,
    gates.T,
    gates.Tdag,
    gates.SWAP,
    gates.ISWAP,
    gates.ISWAPdag,
    gates.SQRTSWAP,
    gates.SQRTISWAPdag,
    gates.SQRTISWAP,
    gates.SQRTISWAPdag,
    gates.BERKELEY,
    gates.BERKELEYdag,
    U1,
]

PARAMETRIC_GATE = [
    gates.RX(0.5),
    gates.RY(0.5),
    gates.RZ(0.5),
    gates.PHASE(0.5),
    gates.R(0.5, 0.9),
    gates.QASMU(0.1, 0.2, 0.3),
    gates.SWAPALPHA(0.3),
    gates.MS(0.47, 0.8),
    gates.RZX(0.6),
    U2(0.447),
]

CONTROLLED_GATE = [
    gates.CX,
    gates.CY,
    gates.CZ,
    gates.CH,
    gates.CS,
    gates.CT,
    gates.CRX(0.7),
    gates.CRY(0.88),
    gates.CRZ(0.78),
    gates.CPHASE(0.9),
    gates.CQASMU(0.9, 0.22, 0.15),
    gates.TOFFOLI,
    gates.FREDKIN,
    controlled(U1, 1, 1),
    controlled(U2, 1, 1)(0.88),
]


@pytest.mark.parametrize("gate", GATES + PARAMETRIC_GATE + CONTROLLED_GATE)
def test_gate_inverse(gate: Unitary | type[Unitary]):
    n = 1 << gate.num_qubits
    inverse = gate.inverse()
    np.testing.assert_allclose(
        (gate.get_qobj() * inverse.get_qobj()).full(),
        np.eye(n),
        atol=1e-12,
    )


def test_gate_equality():
    assert gates.CX == gates.CX
    assert gates.CX != gates.CY
    assert gates.CX != gates.CRX(0.5)

    assert gates.RX(0.5) == gates.RX(0.5)
    assert gates.RX(0.5) != gates.RY(0.5)
    assert gates.RX(0.5) != gates.RX(0.6)

    assert gates.CRX(0.5) == gates.CRX(0.5)
    assert gates.CRX(0.5) != gates.CRY(0.5)
    assert gates.CRX(0.5) != gates.CRX(0.6)


class TestInverse:
    def test_closed_form_and_self_inverse(self):
        assert gates.T.inverse() is Inverse(gates.T) is gates.Tdag
        assert gates.X.inverse() is Inverse(gates.X) is gates.X
        assert gates.RX(0.5).inverse() == Inverse(gates.RX(0.5)) == gates.RX(-0.5)

    def test_wrapper_when_there_is_no_closed_form(self):
        Udag = U1.inverse()
        assert isinstance(Udag, Inverse) and Inverse(U1) == Udag
        assert Udag.inverse() is Inverse(Udag) is U1  # U.inverse().inverse()
        assert Udag.name == "U1dag" and Udag.is_clifford is False
        assert (Udag.num_qreg, Udag.qreg_dim) == (1, (2,))
        assert Udag.get_qobj() == U1.get_qobj().dag()
        assert pickle.loads(pickle.dumps(Udag)) == Udag

    def test_controlled_inverse_keeps_its_controls(self):
        CUdag = controlled(U1).inverse()
        assert CUdag == controlled(U1.inverse()) == Inverse(controlled(U1))
        assert isinstance(CUdag, ControlledUnitary) and CUdag.num_ctrl_qreg == 1
        assert CUdag.name == "CU1dag"
        assert CUdag.get_qobj() == controlled(U1).get_qobj().dag()
        assert gates.CT.inverse() is gates.CTdag
        assert gates.CRX(0.5).inverse() == gates.CRX(-0.5)

    def test_errors(self):
        with pytest.raises(TypeError):
            gates.RX.inverse()  # RX(0.5).inverse is valid

        with pytest.raises(TypeError):
            Inverse(Mz)  # non-unitary

        with pytest.raises(TypeError):

            class OverridesInverse(Unitary):
                num_qreg = 1

                def inverse(self): ...  # the method to overwrite is compute_inverse


class TestControlled:
    def test_factory_and_hand_written_classes_are_one_value(self):
        assert controlled(gates.X) == gates.CX
        assert hash(controlled(gates.X)) == hash(gates.CX)
        assert controlled(gates.X, 2) == gates.TOFFOLI
        assert controlled(gates.SWAP) == gates.FREDKIN
        assert controlled(gates.RX)(0.5) == controlled(gates.RX(0.5)) == gates.CRX(0.5)
        assert pickle.loads(pickle.dumps(controlled(gates.X, 2))) is controlled(
            gates.X, 2
        )
        assert pickle.loads(pickle.dumps(controlled(gates.RX)(0.5))) == gates.CRX(0.5)

    def test_nested_controls_flatten(self):
        assert controlled(gates.CX) == controlled(gates.X, 2) == gates.TOFFOLI
        assert controlled(gates.CX).target is gates.X
        nested = controlled(controlled(gates.X, ctrl_value=0))
        assert nested == controlled(gates.X, 2, ctrl_value=0b10)

    def test_registers_follow_the_target(self):
        assert (controlled(gates.RX).num_qreg, controlled(gates.RX).qreg_dim) == (
            2,
            (2, 2),
        )
        assert controlled(gates.X, ctrl_dim=(3,)).qreg_dim == (3, 2)
        assert controlled(gates.X).self_inverse and not controlled(gates.T).self_inverse

        class Identity(Unitary):
            n: int
            num_qreg = property(lambda self: self.n)

            def compute_qobj(self, dtype="dense"):
                return qutip.qeye([2] * self.n).to(dtype)

        CI = controlled(Identity)(2)
        assert (CI.num_qreg, CI.qreg_dim) == (3, (2, 2, 2))
        assert CI.get_qobj() == qutip.qeye([2, 2, 2])

    def test_target_parameters_and_control_value(self):
        assert gates.CRX(0.5).params == (0.5,)
        assert gates.CRX(0.5).target_unitary == gates.RX(0.5)
        with pytest.raises(TypeError):
            gates.CRX(True)  # the target validates its own parameters

    def test_errors(self):
        with pytest.raises(TypeError):
            controlled(Mz)  # only unitaries have controlled forms
        with pytest.raises(ValueError):
            controlled(gates.X, ctrl_dim=(2, 2))  # one control, two dimensions

    def test_nested_value_keeps_its_parameters(self):
        class CCRXh(ControlledUnitary):
            target = gates.CRX(0.5)

        assert CCRXh == controlled(gates.CRX(0.5))


class TestGateErrors:
    def test_gateclass_errors(self):
        with pytest.raises(ValueError):

            class BadGate(Unitary):
                num_qreg = -1

                def compute_qobj(cls):
                    pass

        with pytest.raises(TypeError):

            class BadGate2(Unitary):
                num_qreg = 1
                is_clifford = 1  # attribute 'is_clifford' must be a bool

                def compute_qobj(dtype):
                    pass

        with pytest.raises(TypeError):

            class BadGate3(Unitary):
                num_qreg = 1
                self_inverse = 1  # attribute 'self_inverse' must be a bool

                def compute_qobj(dtype):
                    pass

        class GoodGate(Unitary):
            num_qreg = 1

            @staticmethod
            def compute_qobj(dtype):
                pass

        with pytest.raises(AttributeError):
            # For a given gateclass, class attribute like num_qubit can't be modified
            GoodGate.num_qreg = 2

        U_rect = qutip.Qobj(np.eye(2, 3))
        with pytest.raises(ValueError):
            get_unitary_gate("U_rect", U_rect)  # U must be a square matrix

        U_not_unitary = qutip.Qobj(np.zeros((2, 2)))
        with pytest.raises(ValueError):
            get_unitary_gate("U_not_unitary", U_not_unitary)  # U must be unitaru

        class GoodParamGate(AngleUnitary):
            num_qreg = 1
            theta: float
            phi: float

            def compute_qobj(self, dtype):
                return qutip.qeye(2)

        with pytest.raises(TypeError):
            GoodParamGate(1.0)

    def test_controlled_gate_errors(self):
        with pytest.raises(ValueError):

            class BadCtrlGate(ControlledUnitary):
                target = gates.X
                num_ctrl_qreg = -1  # Can't be negative
                ctrl_value = 1

        with pytest.raises(ValueError):

            class BadCtrlGate2(ControlledUnitary):
                target = gates.X
                num_ctrl_qreg = 1
                ctrl_value = 1.5  # Control value must be an int

        with pytest.raises(ValueError):

            class BadCtrlGate3(ControlledUnitary):
                target = gates.X
                num_ctrl_qreg = 1
                ctrl_value = 2  # Control value can't be greater than 1 in this case

    def test_utils_errors(self):
        from qutip_qip.operations.utils import (
            _check_oper_dims,
            _targets_to_list,
            expand_operator,
            gate_sequence_product,
        )

        with pytest.raises(ValueError):
            _check_oper_dims(
                qutip.basis(2, 0)
            )  # The operator is not an Qobj with the same input and output dimensions.

        op = qutip.qeye(2)
        with pytest.raises(ValueError):
            _check_oper_dims(oper=op, dims=[3], targets=[0])  # The dims don't match

        with pytest.raises(TypeError):
            _targets_to_list(
                1.5, op, 1
            )  # targets should be an integer or a list of integer

        with pytest.raises(ValueError):
            expand_operator(op, 2, 0)  # dims needs to be an iterable

        with pytest.raises(ValueError):
            gate_sequence_product([], left_to_right=True)  # empty U_list

    def test_standard_gates_failures(self):
        # Instantiation on non-parametric
        with pytest.raises(TypeError):
            gates.CX()

        # Wrong no. of parameters
        with pytest.raises(TypeError):
            gates.CRX(0, 1, 2)

        # Wrong no. of arguments on parametric
        with pytest.raises(TypeError):
            gates.CRX.get_qobj()

        # Control_value > 2^n -1
        with pytest.raises(ValueError):
            controlled(gates.X, num_ctrl_qreg=1, ctrl_value=2)

        with pytest.raises(ValueError):
            controlled(gates.X, num_ctrl_qreg=0)  # num_ctrl_qreg > 0

    def test_class_attribute_modification(self):
        with pytest.raises(AttributeError):
            gates.X.num_qreg = 0

        with pytest.raises(AttributeError):
            gates.X.is_clifford = False

        with pytest.raises(AttributeError):
            gates.X.self_inverse = False

        with pytest.raises(AttributeError):
            gates.CX.num_ctrl_qreg = 0

        with pytest.raises(AttributeError):
            gates.CX.ctrl_value = 0

        with pytest.raises(AttributeError):
            gates.CX.target = gates.Y

        with pytest.raises(AttributeError):
            gates.RY.latex_str = "R_y"

        class H(Unitary):
            num_qreg = 1

            def compute_qobj(dtype):
                pass

        assert H.name == "H"
        assert H.latex_str == "H"


class TestCachedMethod:
    def test_cache_is_per_value_and_per_argument(self):
        dense, csr = gates.X.get_qobj(), gates.X.get_qobj(dtype="csr")
        assert csr.dtype is qutip.data.CSR
        assert gates.X.get_qobj() is dense
        assert gates.X.get_qobj(dtype="csr") is csr
        assert gates.Y.get_qobj() is not dense
