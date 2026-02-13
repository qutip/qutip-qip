import pytest
import os
import shutil
import numpy as np
from pathlib import Path


from qutip_qip.circuit import QubitCircuit, CircuitSimulator
from qutip_qip.circuit.draw import TeXRenderer
from qutip import (
    tensor,
    Qobj,
    ptrace,
    rand_ket,
    fock_dm,
    basis,
    bell_state,
    ket2dm,
    identity,
)
from qutip_qip.qasm import read_qasm
from qutip_qip.operations import Gate, Measurement, gate_sequence_product
import qutip_qip.operations.std as std
from qutip_qip.transpiler import to_chain_structure
from qutip_qip.decompose.decompose_single_qubit_gate import _ZYZ_rotation

import qutip as qp


def _op_dist(A, B):
    return (A - B).norm()


def _teleportation_circuit():
    teleportation = QubitCircuit(
        3, num_cbits=2, input_states=["q0", "0", "0", "c0", "c1"]
    )

    teleportation.add_gate(std.H, targets=[1])
    teleportation.add_gate(std.CX, targets=[2], controls=[1])
    teleportation.add_gate(std.CX, targets=[1], controls=[0])
    teleportation.add_gate(std.H, targets=[0])
    teleportation.add_measurement("M0", targets=[0], classical_store=1)
    teleportation.add_measurement("M1", targets=[1], classical_store=0)
    teleportation.add_gate(std.X, targets=[2], classical_controls=[0])
    teleportation.add_gate(std.Z, targets=[2], classical_controls=[1])

    return teleportation


def _teleportation_circuit2():
    teleportation = QubitCircuit(
        3, num_cbits=2, input_states=["q0", "0", "0", "c0", "c1"]
    )

    teleportation.add_gate(std.H, targets=[1])
    teleportation.add_gate(std.CX, targets=[2], controls=[1])
    teleportation.add_gate(std.CX, targets=[1], controls=[0])
    teleportation.add_gate(std.H, targets=[0])
    teleportation.add_gate(std.CX, targets=[2], controls=[1])
    teleportation.add_gate(std.CZ, targets=[2], controls=[0])

    return teleportation


def _measurement_circuit():
    qc = QubitCircuit(2, num_cbits=2)

    qc.add_measurement("M0", targets=[0], classical_store=0)
    qc.add_measurement("M1", targets=[1], classical_store=1)

    return qc


class TestQubitCircuit:
    """
    A test class for the QuTiP functions for Circuit resolution.
    """

    @pytest.mark.parametrize(
        ["gate_from", "gate_to", "targets", "controls"],
        [
            pytest.param(std.SWAP, "CX", [0, 1], [], id="SWAPtoCX"),
            pytest.param(std.ISWAP, "CX", [0, 1], [], id="ISWAPtoCX"),
            pytest.param(std.CZ, "CX", [1], [0], id="CZtoCX"),
            pytest.param(std.CX, "CZ", [0], [1], id="CXtoCZ"),
            pytest.param(std.CX, "SQRTSWAP", [0], [1], id="CXtoSQRTSWAP"),
            pytest.param(std.CX, "SQRTISWAP", [0], [1], id="CXtoSQRTISWAP"),
            pytest.param(std.CX, "ISWAP", [0], [1], id="CXtoISWAP"),
        ],
    )
    def testresolve(self, gate_from, gate_to, targets, controls):
        qc1 = QubitCircuit(2)
        qc1.add_gate(gate_from, targets=targets, controls=controls)
        U1 = qc1.compute_unitary()
        qc2 = qc1.resolve_gates(basis=gate_to)
        U2 = qc2.compute_unitary()
        assert _op_dist(U1, U2) < 1e-12

    def testHdecompose(self):
        """
        H to rotation: compare unitary matrix for H and product of
        resolved matrices in terms of rotation gates.
        """
        qc1 = QubitCircuit(1)
        qc1.add_gate(std.H, targets=0)
        U1 = qc1.compute_unitary()
        qc2 = qc1.resolve_gates()
        U2 = qc2.compute_unitary()
        assert _op_dist(U1, U2) < 1e-12

    def testFREDKINdecompose(self):
        """
        FREDKIN to rotation and CNOT: compare unitary matrix for FREDKIN and product of
        resolved matrices in terms of rotation gates and CNOT.
        """
        qc1 = QubitCircuit(3)
        qc1.add_gate(std.FREDKIN, targets=[0, 1], controls=[2])
        U1 = qc1.compute_unitary()
        qc2 = qc1.resolve_gates()
        U2 = qc2.compute_unitary()
        assert _op_dist(U1, U2) < 1e-12

    def test_add_gate(self):
        """
        Addition of a gate object directly to a `QubitCircuit`
        """
        qc = QubitCircuit(6)
        qc.add_gate(std.CX, targets=[1], controls=[0])
        qc.add_gate(std.SWAP, targets=[1, 4])
        qc.add_gate(std.TOFFOLI, controls=[0, 1], targets=[2])
        qc.add_gate(std.H, targets=[3])
        qc.add_gate(std.SWAP, targets=[1, 4])
        qc.add_gate(std.RY(arg_value=1.570796), targets=4)
        qc.add_gate(std.RY(arg_value=1.570796), targets=5)
        qc.add_gate(std.RX(arg_value=-1.570796), targets=[3])

        # Test explicit gate addition
        assert qc.instructions[0].operation.name == "CX"
        assert qc.instructions[0].targets == (1,)
        assert qc.instructions[0].controls == (0,)

        # Test direct gate addition
        assert qc.instructions[1].operation.name == "SWAP"
        assert qc.instructions[1].targets == (1, 4)

        # Test specified position gate addition
        assert qc.instructions[3].operation.name == "H"
        assert qc.instructions[3].targets == (3,)

        # Test adding 1 qubit gate on [start, end] qubits
        assert qc.instructions[5].operation.name == "RY"
        assert qc.instructions[5].targets == (4,)
        assert qc.instructions[5].operation.arg_value[0] == 1.570796
        assert qc.instructions[6].operation.name == "RY"
        assert qc.instructions[6].targets == (5,)
        assert qc.instructions[6].operation.arg_value[0] == 1.570796

        # Test adding 1 qubit gate on qubits [3]
        assert qc.instructions[7].operation.name == "RX"
        assert qc.instructions[7].targets == (3,)
        assert qc.instructions[7].operation.arg_value[0] == -1.570796

        class DUMMY1(Gate):
            num_qubits = 1
            self_inverse = False

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def get_qobj(self):
                pass

        class DUMMY2(DUMMY1):
            pass

        qc.add_gate(DUMMY1, targets=0)
        qc.add_gate(DUMMY2, targets=0)

        # Test adding gates at multiple (sorted) indices at once.
        # NOTE: Every insertion shifts the indices in the original list of
        #       gates by an additional position to the right.
        expected_gate_names = [
            "CX",
            "SWAP",
            "TOFFOLI",
            "H",
            "SWAP",
            "RY",
            "RY",
            "RX",
            "DUMMY1",
            "DUMMY2",
        ]
        actual_gate_names = [ins.operation.name for ins in qc.instructions]
        assert actual_gate_names == expected_gate_names

    def test_add_circuit(self):
        """
        Addition of a circuit to a `QubitCircuit`
        """

        qc = QubitCircuit(6)
        qc.add_gate(std.CX, targets=[1], controls=[0])
        qc.add_gate(std.SWAP, targets=[1, 4])
        qc.add_gate(std.TOFFOLI, controls=[0, 1], targets=[2])
        qc.add_gate(std.H, targets=[3])
        qc.add_gate(std.SWAP, targets=[1, 4])
        qc.add_measurement("M0", targets=[0], classical_store=[1])
        qc.add_gate(std.RY(1.570796), targets=4)
        qc.add_gate(std.RY(1.570796), targets=5)
        qc.add_gate(std.CRX(np.pi / 2), controls=[1], targets=[2])

        qc1 = QubitCircuit(6)
        qc1.add_circuit(qc)

        # Test if all gates and measurements are added
        assert len(qc1.instructions) == len(qc.instructions)

        for i in range(len(qc1.instructions)):
            assert (
                qc1.instructions[i].operation.name
                == qc.instructions[i].operation.name
            )
            assert (
                qc1.instructions[i].qubits[0] == qc.instructions[i].qubits[0]
            )
            if qc1.instructions[i].is_gate_instruction() and (
                qc.instructions[i].is_gate_instruction()
            ):
                if qc.instructions[i].operation.is_controlled_gate():
                    assert (
                        qc1.instructions[i].controls
                        == qc.instructions[i].controls
                    )
                assert (
                    qc1.instructions[i].cbits_ctrl_value
                    == qc.instructions[i].cbits_ctrl_value
                )
            elif qc1.instructions[i].is_measurement_instruction() and qc.instructions[i].is_measurement_instruction():
                assert qc1.instructions[i].cbits == qc.instructions[i].cbits

        # Test exception when qubit out of range
        pytest.raises(NotImplementedError, qc1.add_circuit, qc, start=4)

        qc2 = QubitCircuit(8)
        qc2.add_circuit(qc, start=2)

        # Test if all gates are added
        assert len(qc2.instructions) == len(qc.instructions)

        # Test if the positions are correct
        for i in range(len(qc2.instructions)):
            if qc.instructions[i].is_gate_instruction():
                assert (
                    qc2.instructions[i].targets[0]
                    == qc.instructions[i].targets[0] + 2
                )
                if qc.instructions[i].operation.is_controlled_gate():
                    assert (
                        qc2.instructions[i].controls[0]
                        == qc.instructions[i].controls[0] + 2
                    )

    def test_add_state(self):
        """
        Addition of input and output states to a circuit.
        """
        qc = QubitCircuit(3)

        qc.add_state("0", targets=[0])
        qc.add_state("+", targets=[1], state_type="output")
        qc.add_state("-", targets=[1])

        assert qc.input_states[0] == "0"
        assert qc.input_states[2] is None
        assert qc.output_states[1] == "+"

        qc1 = QubitCircuit(10)

        qc1.add_state("0", targets=[2, 3, 5, 6])
        qc1.add_state("+", targets=[1, 4, 9])
        qc1.add_state("A", targets=[1, 4, 9], state_type="output")
        qc1.add_state("A", targets=[1, 4, 9], state_type="output")
        qc1.add_state("beta", targets=[0], state_type="output")
        assert qc1.input_states[0] is None

        assert qc1.input_states[2] == "0"
        assert qc1.input_states[3] == "0"
        assert qc1.input_states[6] == "0"
        assert qc1.input_states[1] == "+"
        assert qc1.input_states[4] == "+"

        assert qc1.output_states[2] is None
        assert qc1.output_states[1] == "A"
        assert qc1.output_states[4] == "A"
        assert qc1.output_states[9] == "A"

        assert qc1.output_states[0] == "beta"

    def test_add_measurement(self):
        """
        Addition of Measurement Object to a circuit.
        """

        qc = QubitCircuit(3, num_cbits=3)

        qc.add_measurement("M0", targets=[0], classical_store=0)
        qc.add_gate(std.CX, targets=[1], controls=[0])
        qc.add_gate(std.TOFFOLI, controls=[0, 1], targets=[2])
        qc.add_measurement("M1", targets=[2], classical_store=1)
        qc.add_gate(std.H, targets=[1], classical_controls=[0, 1])
        qc.add_measurement("M2", targets=[1], classical_store=2)

        # checking correct addition of measurements
        assert qc.instructions[0].qubits[0] == 0
        assert qc.instructions[0].cbits[0] == 0
        assert qc.instructions[3].operation.name == "M1"
        assert qc.instructions[5].cbits[0] == 2

        # checking if gates are added correctly with measurements
        assert qc.instructions[2].operation.name == "TOFFOLI"
        assert qc.instructions[4].cbits == (0, 1)

    @pytest.mark.skip(reason="Changing the interface completely")
    @pytest.mark.parametrize("gate", ["X", "Y", "Z", "S", "T"])
    def test_exceptions(self, gate):
        """
        Text exceptions are thrown correctly for inadequate inputs
        """
        qc = QubitCircuit(2)
        pytest.raises(ValueError, qc.add_gate, gate, targets=[1], controls=[0])

    def test_single_qubit_gates(self):
        """
        Text single qubit gates are added correctly
        """
        qc = QubitCircuit(3)

        qc.add_gate(std.X, targets=[0])
        qc.add_gate(std.CY, targets=[1], controls=[0])
        qc.add_gate(std.Y, targets=[2])
        qc.add_gate(std.CS, targets=[0], controls=[1])
        qc.add_gate(std.Z, targets=[1])
        qc.add_gate(std.CT, targets=[1], controls=[2])
        qc.add_gate(std.CZ, targets=[0], controls=[1])
        qc.add_gate(std.S, targets=[1])
        qc.add_gate(std.T, targets=[2])

        assert qc.instructions[8].operation.name == "T"
        assert qc.instructions[7].operation.name == "S"
        assert qc.instructions[6].operation.name == "CZ"
        assert qc.instructions[5].operation.name == "CT"
        assert qc.instructions[4].operation.name == "Z"
        assert qc.instructions[3].operation.name == "CS"
        assert qc.instructions[2].operation.name == "Y"
        assert qc.instructions[1].operation.name == "CY"
        assert qc.instructions[0].operation.name == "X"

        assert qc.instructions[8].targets == (2,)
        assert qc.instructions[7].targets == (1,)
        assert qc.instructions[6].targets == (0,)
        assert qc.instructions[5].targets == (1,)
        assert qc.instructions[4].targets == (1,)
        assert qc.instructions[3].targets == (0,)
        assert qc.instructions[2].targets == (2,)
        assert qc.instructions[1].targets == (1,)
        assert qc.instructions[0].targets == (0,)

        assert qc.instructions[6].controls == (1,)
        assert qc.instructions[5].controls == (2,)
        assert qc.instructions[3].controls == (1,)
        assert qc.instructions[1].controls == (0,)

    def test_reverse(self):
        """
        Reverse a quantum circuit
        """
        qc = QubitCircuit(3, num_cbits=1)

        qc.add_gate(std.RX(arg_value=3.141, arg_label=r"\pi/2"), targets=[0])
        qc.add_gate(std.CX, targets=[1], controls=[0])
        qc.add_measurement("M1", targets=[1], classical_store=0)
        qc.add_gate(std.H, targets=[2])
        # Keep input output same

        qc.add_state("0", targets=[0])
        qc.add_state("+", targets=[1], state_type="output")
        qc.add_state("-", targets=[1])

        qc_rev = qc.reverse_circuit()

        assert qc_rev.instructions[0].operation.name == "H"
        assert qc_rev.instructions[1].operation.name == "M1"
        assert qc_rev.instructions[2].operation.name == "CX"
        assert qc_rev.instructions[3].operation.name == "RX"

        assert qc_rev.input_states[0] == "0"
        assert qc_rev.input_states[2] is None
        assert qc_rev.output_states[1] == "+"

    def test_user_gate(self):
        """
        User defined gate for QubitCircuit
        """

        def customer_gate1(arg_values):
            mat = np.zeros((4, 4), dtype=np.complex128)
            mat[0, 0] = mat[1, 1] = 1.0
            mat[2:4, 2:4] = std.RX(arg_values).get_qobj().full()
            return Qobj(mat, dims=[[2, 2], [2, 2]])

        class T1(Gate):
            num_qubits = 1
            self_inverse = True

            def __init__(self):
                pass

            @staticmethod
            def get_qobj():
                mat = np.array([[1.0, 0], [0.0, 1.0j]])
                return Qobj(mat, dims=[[2], [2]])

        qc = QubitCircuit(3)
        qc.add_gate(std.CRX(np.pi / 2), targets=[2], controls=[1])
        qc.add_gate(T1, targets=[1])
        props = qc.propagators()
        result1 = tensor(identity(2), customer_gate1(np.pi / 2))
        np.testing.assert_allclose(props[0].full(), result1.full())
        result2 = tensor(identity(2), T1.get_qobj(), identity(2))
        np.testing.assert_allclose(props[1].full(), result2.full())

    def test_N_level_system(self):
        """
        Test for circuit with N-level system.
        """
        mat3 = qp.rand_unitary(3)

        class CTRLMAT3(Gate):
            num_qubits = 2
            self_inverse = False

            def validate_params(self):
                pass

            @staticmethod
            def get_qobj():
                """
                A qubit control an operator acting on a 3 level system
                """
                dim = mat3.dims[0][0]
                return tensor(fock_dm(2, 1), mat3) + tensor(
                    fock_dm(2, 0), identity(dim)
                )

        qc = QubitCircuit(2, dims=[3, 2])
        qc.add_gate(CTRLMAT3, targets=[1, 0])
        props = qc.propagators()
        final_fid = qp.average_gate_fidelity(mat3, ptrace(props[0], 0) - 1)
        assert pytest.approx(final_fid, 1.0e-6) == 1

        init_state = basis([3, 2], [0, 1])
        result = qc.run(init_state)
        final_fid = qp.fidelity(result, props[0] * init_state)
        assert pytest.approx(final_fid, 1.0e-6) == 1.0

    @pytest.mark.repeat(10)
    def test_run_teleportation(self):
        """
        Test circuit run and mid-circuit measurement functionality
        by repeating the teleportation circuit on multiple random kets
        """

        teleportation = _teleportation_circuit()

        state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))
        initial_measurement = Measurement("start", targets=[0])
        _, initial_probabilities = initial_measurement.measurement_comp_basis(
            state
        )

        teleportation_sim = CircuitSimulator(teleportation)
        teleportation_sim_results = teleportation_sim.run(state)
        state_final = teleportation_sim_results.get_final_states(0)

        final_measurement = Measurement("start", targets=[2])
        _, final_probabilities = final_measurement.measurement_comp_basis(
            state_final
        )

        np.testing.assert_allclose(initial_probabilities, final_probabilities)

    def test_classical_control(self):
        qc = QubitCircuit(1, num_cbits=2)
        qc.add_gate(
            std.X,
            targets=[0],
            classical_controls=[0, 1],
            classical_control_value=1,
        )
        result = qc.run(basis(2, 0), cbits=[1, 0])
        fid = qp.fidelity(result, basis(2, 0))
        assert pytest.approx(fid, 1.0e-6) == 1

        qc = QubitCircuit(1, num_cbits=2)
        qc.add_gate(
            std.X,
            targets=[0],
            classical_controls=[0, 1],
            classical_control_value=2,
        )
        result = qc.run(basis(2, 0), cbits=[1, 0])
        fid = qp.fidelity(result, basis(2, 1))
        assert pytest.approx(fid, 1.0e-6) == 1

    def test_runstatistics_teleportation(self):
        """
        Test circuit run_statistics on teleportation circuit
        """

        teleportation = _teleportation_circuit()
        final_measurement = Measurement("start", targets=[2])
        initial_measurement = Measurement("start", targets=[0])

        original_state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))
        _, initial_probabilities = initial_measurement.measurement_comp_basis(
            original_state
        )

        teleportation_results = teleportation.run_statistics(original_state)
        states = teleportation_results.get_final_states()
        probabilities = teleportation_results.get_probabilities()

        for i, state in enumerate(states):
            state_final = state
            prob = probabilities[i]
            _, final_probabilities = final_measurement.measurement_comp_basis(
                state_final
            )
            np.testing.assert_allclose(
                initial_probabilities, final_probabilities
            )
            assert prob == pytest.approx(0.25, abs=1e-7)

        mixed_state = sum(p * ket2dm(s) for p, s in zip(probabilities, states))
        dm_state = ket2dm(original_state)

        teleportation2 = _teleportation_circuit2()

        final_state = teleportation2.run(dm_state)
        _, probs1 = final_measurement.measurement_comp_basis(final_state)
        _, probs2 = final_measurement.measurement_comp_basis(mixed_state)

        np.testing.assert_allclose(probs1, probs2)

    def test_measurement_circuit(self):

        qc = _measurement_circuit()
        simulator = CircuitSimulator(qc)
        labels = ["00", "01", "10", "11"]

        for label in labels:
            state = bell_state(label)
            simulator.run(state)
            if label[0] == "0":
                assert simulator.cbits[0] == simulator.cbits[1]
            else:
                assert simulator.cbits[0] != simulator.cbits[1]

    def test_circuit_with_selected_measurement_result(self):
        qc = QubitCircuit(num_qubits=1, num_cbits=1)
        qc.add_gate(std.H, targets=0)
        qc.add_measurement("M0", targets=0, classical_store=0)

        # We reset the random seed so that
        # if we don's select the measurement result,
        # the two circuit should return the same value.
        np.random.seed(0)
        final_state = qc.run(qp.basis(2, 0), cbits=[0], measure_results=[0])
        fid = pytest.approx(qp.fidelity(final_state, basis(2, 0)))
        assert fid == 1.0
        np.random.seed(0)
        final_state = qc.run(qp.basis(2, 0), cbits=[0], measure_results=[1])
        fid = pytest.approx(qp.fidelity(final_state, basis(2, 1)))
        assert fid == 1.0

    def test_gate_product(self):

        filename = "qft.qasm"
        filepath = Path(__file__).parent / "qasm_files" / filename
        qc = read_qasm(filepath)

        U_list_expanded = qc.propagators()
        U_list = qc.propagators(expand=False)

        inds_list = []

        for circ_instruction in qc.instructions:
            if circ_instruction.is_measurement_instruction():
                continue
            else:
                inds_list.append(circ_instruction.qubits)

        U_1, _ = gate_sequence_product(
            U_list, inds_list=inds_list, expand=True
        )
        U_2 = gate_sequence_product(
            U_list_expanded, left_to_right=True, expand=False
        )

        np.testing.assert_allclose(U_1.full(), U_2.full())

    def test_wstate(self):

        filename = "w-state.qasm"
        filepath = Path(__file__).parent / "qasm_files" / filename
        qc = read_qasm(filepath)

        rand_state = rand_ket(2)
        state = tensor(basis(2, 0), basis(2, 0), basis(2, 0), rand_state)

        fourth = Measurement("test_rand", targets=[3])

        _, probs_initial = fourth.measurement_comp_basis(state)

        simulator = CircuitSimulator(qc)

        result = simulator.run_statistics(state)
        final_states = result.get_final_states()
        result_cbits = result.get_cbits()

        for i, final_state in enumerate(final_states):
            _, probs_final = fourth.measurement_comp_basis(final_state)
            np.testing.assert_allclose(probs_initial, probs_final)
            assert sum(result_cbits[i]) == 1

    _latex_template = r"""
\documentclass[border=3pt]{standalone}
\usepackage[braket]{qcircuit}
\begin{document}
\Qcircuit @C=1cm @R=1cm {
%s}
\end{document}
"""

    def test_latex_code_teleportation_circuit(self):
        qc = _teleportation_circuit()
        renderer = TeXRenderer(qc)
        latex = renderer.latex_code()
        assert latex == renderer._latex_template % "\n".join(
            [
                r" & \lstick{c1} &  \qw  &  \qw  &  \qw  &  \qw"
                r"  &  \qw \cwx[4]  &  \qw  &  \qw  &  \ctrl{2}  & \qw \\ ",
                r" & \lstick{c0} &  \qw  &  \qw  &  \qw  &  \qw"
                r"  &  \qw  &  \qw \cwx[2]  &  \ctrl{1}  &  \qw  & \qw \\ ",
                r" & \lstick{\ket{0}} &  \qw  &  \targ  &  \qw  &  \qw"
                r"  &  \qw  &  \qw  &  \gate{X}  &  \gate{Z}  & \qw \\ ",
                r" & \lstick{\ket{0}} &  \gate{H}  &  \ctrl{-1}  &"
                r"  \targ  &  \qw  &  \qw  &  \meter &  \qw  &  \qw  & \qw \\ ",
                r" & \lstick{\ket{q0}} &  \qw  &  \qw  &  \ctrl{-1}  &"
                r"  \gate{H}  &  \meter &  \qw  &  \qw  &  \qw  & \qw \\ ",
                "",
            ]
        )

    def test_latex_code_classical_controls(self):
        qc = QubitCircuit(1, num_cbits=1, reverse_states=True)
        qc.add_gate(std.X, targets=0, classical_controls=[0])
        renderer = TeXRenderer(qc)
        latex = TeXRenderer(qc).latex_code()
        assert latex == renderer._latex_template % "\n".join(
            [
                r" &  &  \ctrl{1}  & \qw \\ ",
                r" &  &  \gate{X}  & \qw \\ ",
                "",
            ]
        )

        qc = QubitCircuit(1, num_cbits=1, reverse_states=False)
        qc.add_gate(std.X, targets=0, classical_controls=[0])
        renderer = TeXRenderer(qc)
        latex = TeXRenderer(qc).latex_code()
        assert latex == renderer._latex_template % "\n".join(
            [
                r" &  &  \gate{X}  & \qw \\ ",
                r" &  &  \ctrl{-1}  & \qw \\ ",
                "",
            ]
        )

    H = Qobj(
        [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
    )
    H_zyz_gates = _ZYZ_rotation(H)
    H_zyz_quantum_circuit = QubitCircuit(1)
    for g in H_zyz_gates:
        H_zyz_quantum_circuit.add_gate(g, targets=[0])  # TODO CHECK
    sigmax_zyz_gates = _ZYZ_rotation(std.X.get_qobj())
    sigmax_zyz_quantum_circuit = QubitCircuit(1)
    for g in sigmax_zyz_gates:
        sigmax_zyz_quantum_circuit.add_gate(g, targets=[0])

    @pytest.mark.parametrize(
        "valid_input, correct_result",
        [(H_zyz_quantum_circuit, H), (sigmax_zyz_quantum_circuit, std.X.get_qobj())],
    )
    def test_compute_unitary(self, valid_input, correct_result):
        final_output = valid_input.compute_unitary()
        assert isinstance(final_output, Qobj)
        assert final_output == correct_result

    def test_latex_code(self):
        qc = QubitCircuit(1, num_cbits=1, reverse_states=True)
        qc.add_measurement("M0", targets=0, classical_store=0)
        exp = (
            " &  &  \\qw \\cwx[1]  & \\qw \\\\ \n &  &  \\meter & \\qw \\\\ \n"
        )

        renderer = TeXRenderer(qc)
        assert renderer.latex_code() == renderer._latex_template % exp

    def test_latex_code_non_reversed(self):
        qc = QubitCircuit(1, num_cbits=1, reverse_states=False)
        qc.add_measurement("M0", targets=0, classical_store=0)
        exp = (
            " &  &  \\meter & \\qw \\\\ \n &  "
            + "&  \\qw \\cwx[-1]  & \\qw \\\\ \n"
        )
        renderer = TeXRenderer(qc)
        assert renderer.latex_code() == renderer._latex_template % exp

    @pytest.mark.skipif(
        shutil.which("pdflatex") is None, reason="requires pdflatex"
    )
    def test_export_image(self, in_temporary_directory):
        from qutip_qip.circuit.texrenderer import CONVERTERS

        qc = QubitCircuit(2, reverse_states=False)
        qc.add_gate(std.CZ, controls=[0], targets=[1])

        if "png" in CONVERTERS:
            file_png200 = "exported_pic_200.png"
            file_png400 = "exported_pic_400.png"
            qc.draw("latex", "png", 200, file_png200.split(".")[0], True)
            qc.draw("latex", "png", 400.5, file_png400.split(".")[0], True)
            assert file_png200 in os.listdir(".")
            assert file_png400 in os.listdir(".")
            assert os.stat(file_png200).st_size < os.stat(file_png400).st_size
        if "svg" in CONVERTERS:
            file_svg = "exported_pic.svg"
            qc.draw("svg", file_svg.split(".")[0], True)
            assert file_svg in os.listdir(".")

    def test_circuit_chain_structure(self):
        """
        Test if the transpiler correctly inherit the properties of a circuit.
        """
        qc = QubitCircuit(3, reverse_states=True)
        qc.add_gate(std.CX, targets=[2], controls=[0])
        qc2 = to_chain_structure(qc)

        assert qc2.reverse_states is True
        assert qc2.input_states == [None] * 3
