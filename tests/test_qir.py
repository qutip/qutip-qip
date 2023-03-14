import typing
import pytest
import numpy as np
import random
from numpy.testing import assert_allclose
from qutip_qip.circuit import QubitCircuit

# Will skip tests in this entire file
# if PyQIR is not available.
pqg = pytest.importorskip("pyqir.generator")

pqp = pytest.importorskip("pyqir.parser")
import pyqir.parser

from qutip_qip import qir

T = typing.TypeVar("T")


def _assert_is_single(collection: typing.List[T]) -> T:
    assert len(collection) == 1
    return collection[0]


def _assert_arg_is_qubit(
    arg: pyqir.parser.QirOperand, idx: typing.Optional[int] = None
):
    assert isinstance(arg, pyqir.parser.QirQubitConstant)
    if idx is not None:
        assert arg.id == idx


def _assert_arg_is_result(
    arg: pyqir.parser.QirOperand, idx: typing.Optional[int] = None
):
    assert isinstance(arg, pyqir.parser.QirResultConstant)
    if idx is not None:
        assert arg.id == idx


def _assert_arg_is_double(
    arg: pyqir.parser.QirOperand, angle: typing.Optional[float] = None
):
    assert isinstance(arg, pyqir.parser.QirDoubleConstant)
    if angle is not None:
        np.testing.assert_allclose(arg.value, angle)


def _assert_is_simple_qis_call(
    inst: pyqir.parser.QirInstr, gate_name: str, targets: typing.List[int]
):
    assert isinstance(inst, pyqir.parser.QirQisCallInstr)
    assert inst.func_name == f"__quantum__qis__{gate_name}__body"
    assert len(inst.func_args) == len(targets)
    for target, arg in zip(targets, inst.func_args):
        _assert_arg_is_qubit(arg, target)


def _assert_is_rotation_qis_call(
    inst: pyqir.parser.QirInstr, gate_name: str, angle: float, target: int
):
    assert isinstance(inst, pyqir.parser.QirQisCallInstr)
    assert inst.func_name == f"__quantum__qis__{gate_name}__body"
    assert len(inst.func_args) == 2
    angle_arg, target_arg = inst.func_args
    _assert_arg_is_double(angle_arg, angle)
    _assert_arg_is_qubit(target_arg, target)


def _assert_is_measurement_qis_call(
    inst: pyqir.parser.QirInstr, gate_name: str, target: int, result: int
):
    assert isinstance(inst, pyqir.parser.QirQisCallInstr)
    assert inst.func_name == f"__quantum__qis__{gate_name}__body"
    assert len(inst.func_args) == 2
    target_arg, result_arg = inst.func_args
    _assert_arg_is_qubit(target_arg, target)
    _assert_arg_is_result(result_arg, result)


class TestConverter:
    """
    Test suite that checks that conversions from circuits to QIR produce
    correct QIR modules.

    Note that since literal byte equivalence is not guaranteed, our testing
    strategy will be to use the PyQIR parser package to read back the QIR that
    we export and check for semantic equivalence.
    """

    def test_simple_x_circuit(self):
        """
        Test to check conversion of a circuit
        containing a single qubit gate.
        """
        circuit = QubitCircuit(1)
        circuit.add_gate("X", targets=[0])
        parsed_qir_module: pyqir.parser.QirModule = qir.circuit_to_qir(
            circuit, format=qir.QirFormat.MODULE
        )
        parsed_func = _assert_is_single(parsed_qir_module.entrypoint_funcs)
        assert parsed_func.required_qubits == 1
        assert parsed_func.required_results == 0
        parsed_block = _assert_is_single(parsed_func.blocks)
        inst = _assert_is_single(parsed_block.instructions)
        _assert_is_simple_qis_call(inst, "x", [0])

    def test_simple_cnot_circuit(self):
        """
        Test to check conversion of a circuit
        containing a single qubit gate.
        """
        circuit = QubitCircuit(2)
        circuit.add_gate("CX", targets=[1], controls=[0])
        parsed_qir_module: pyqir.parser.QirModule = qir.circuit_to_qir(
            circuit, format=qir.QirFormat.MODULE
        )
        parsed_func = _assert_is_single(parsed_qir_module.entrypoint_funcs)
        assert parsed_func.required_qubits == 2
        assert parsed_func.required_results == 0
        parsed_block = _assert_is_single(parsed_func.blocks)
        inst = _assert_is_single(parsed_block.instructions)
        _assert_is_simple_qis_call(inst, "cnot", [0, 1])

    def test_simple_rz_circuit(self):
        """
        Test to check conversion of a circuit
        containing a single qubit gate.
        """
        circuit = QubitCircuit(1)
        circuit.add_gate("RZ", targets=[0], arg_value=0.123)
        parsed_qir_module: pyqir.parser.QirModule = qir.circuit_to_qir(
            circuit, format=qir.QirFormat.MODULE
        )
        parsed_func = _assert_is_single(parsed_qir_module.entrypoint_funcs)
        assert parsed_func.required_qubits == 1
        assert parsed_func.required_results == 0
        parsed_block = _assert_is_single(parsed_func.blocks)
        inst = _assert_is_single(parsed_block.instructions)
        _assert_is_rotation_qis_call(inst, "rz", 0.123, 0)

    def test_teleport_circuit(self):
        # NB: this test is a bit detailed, as it checks metadata throughout
        #     control flow in a teleportation circuit.
        circuit = QubitCircuit(3, num_cbits=2)
        msg, here, there = range(3)
        circuit.add_gate("RZ", targets=[msg], arg_value=0.123)
        circuit.add_gate("SNOT", targets=[here])
        circuit.add_gate("CNOT", targets=[there], controls=[here])
        circuit.add_gate("CNOT", targets=[here], controls=[msg])
        circuit.add_gate("SNOT", targets=[msg])
        circuit.add_measurement("Z", targets=[msg], classical_store=0)
        circuit.add_measurement("Z", targets=[here], classical_store=1)
        circuit.add_gate("X", targets=[there], classical_controls=[0])
        circuit.add_gate("Z", targets=[there], classical_controls=[1])

        parsed_qir_module: pyqir.parser.QirModule = qir.circuit_to_qir(
            circuit, format=qir.QirFormat.MODULE
        )
        parsed_func = _assert_is_single(parsed_qir_module.entrypoint_funcs)
        assert parsed_func.required_qubits == 3
        assert parsed_func.required_results == 2
        assert len(parsed_func.blocks) == 7

        def assert_readresult(inst, result: int):
            assert isinstance(inst, pyqir.parser.QirQisCallInstr)
            assert inst.func_name == "__quantum__qis__read_result__body"
            arg = _assert_is_single(inst.func_args)
            _assert_arg_is_result(arg, result)
            return inst.output_name

        entry = parsed_func.blocks[0]
        then = parsed_func.blocks[1]
        else_ = parsed_func.blocks[2]
        continue_ = parsed_func.blocks[3]
        then2 = parsed_func.blocks[4]
        else3 = parsed_func.blocks[5]
        continue4 = parsed_func.blocks[6]

        # Entry block
        # NB: We only check the name of the entry point block, none of the
        #     others names are semantically relevant, and thus can change
        #     without that being a breaking change.
        assert entry.name == "entry"
        _assert_is_rotation_qis_call(entry.instructions[0], "rz", 0.123, msg)
        _assert_is_simple_qis_call(entry.instructions[1], "h", [here])
        _assert_is_simple_qis_call(
            entry.instructions[2], "cnot", [here, there]
        )
        _assert_is_simple_qis_call(entry.instructions[3], "cnot", [msg, here])
        _assert_is_simple_qis_call(entry.instructions[4], "h", [msg])
        _assert_is_measurement_qis_call(entry.instructions[5], "mz", msg, 0)
        _assert_is_measurement_qis_call(entry.instructions[6], "mz", here, 1)
        cond_label = assert_readresult(entry.instructions[7], 0)
        term = entry.terminator
        assert isinstance(term, pyqir.parser.QirCondBrTerminator)
        cond = term.condition
        assert isinstance(cond, pyqir.parser.QirLocalOperand)
        assert cond.name == cond_label
        assert term.true_dest == then.name
        assert term.false_dest == else_.name

        # Then block
        inst = _assert_is_single(then.instructions)
        _assert_is_simple_qis_call(inst, "x", [there])
        term = then.terminator
        assert isinstance(term, pyqir.parser.QirBrTerminator)
        assert term.dest == continue_.name

        # else block
        assert len(else_.instructions) == 0
        term = else_.terminator
        assert isinstance(term, pyqir.parser.QirBrTerminator)
        assert term.dest == continue_.name

        # continue block
        inst = _assert_is_single(continue_.instructions)
        cond_label = assert_readresult(inst, 1)
        term = continue_.terminator
        assert isinstance(term, pyqir.parser.QirCondBrTerminator)
        cond = term.condition
        assert isinstance(cond, pyqir.parser.QirLocalOperand)
        assert cond.name == cond_label
        assert term.true_dest == then2.name
        assert term.false_dest == else3.name

        # then2 block
        inst = _assert_is_single(then2.instructions)
        _assert_is_simple_qis_call(inst, "z", [there])
        term = then2.terminator
        assert isinstance(term, pyqir.parser.QirBrTerminator)
        assert term.dest == continue4.name

        # else3 block
        assert len(else3.instructions) == 0
        term = else3.terminator
        assert isinstance(term, pyqir.parser.QirBrTerminator)
        assert term.dest == continue4.name

        # continue4 block
        assert len(continue4.instructions) == 0
        term = continue4.terminator
        assert isinstance(term, pyqir.parser.QirRetTerminator)
        assert term.operand is None
