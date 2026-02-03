import pytest
from copy import deepcopy

from qutip_qip.circuit import QubitCircuit
from qutip_qip.compiler import Instruction, Scheduler
from qutip_qip.operations import GATE_CLASS_MAP, ControlledGate
from qutip import qeye, tracedist


def _verify_scheduled_circuit(circuit, gate_cycle_indices):
    """
    Compare results between the original and the scheduled circuit.
    The input gate_cycle_indices is the scheduling result,
    i.e., a list of integers denoting the execution cycle of each gate in the circuit.
    """
    result0 = circuit.compute_unitary()
    scheduled_gate = [[] for i in range(max(gate_cycle_indices) + 1)]
    for i, cycles in enumerate(gate_cycle_indices):
        scheduled_gate[cycles].append(circuit.instructions[i])
    circuit.instructions = sum(scheduled_gate, [])
    result1 = circuit.compute_unitary()
    return tracedist(result0 * result1.dag(), qeye(result0.dims[0])) < 1.0e-7


def test_allow_permutation():
    circuit = QubitCircuit(2)
    circuit.add_gate("X", targets=0)
    circuit.add_gate("CNOT", targets=0, controls=1)
    circuit.add_gate("X", targets=1)

    scheduler = Scheduler("ASAP", allow_permutation=True)
    gate_cycle_indices = scheduler.schedule(circuit)
    assert (max(gate_cycle_indices) + 1) == 2

    scheduler = Scheduler("ASAP", allow_permutation=False)
    gate_cycle_indices = scheduler.schedule(circuit)
    assert (max(gate_cycle_indices) + 1) == 3


def _circuit_cnot_x():
    circuit = QubitCircuit(2)
    circuit.add_gate("X", targets=0)
    circuit.add_gate("CNOT", targets=0, controls=1)
    circuit.add_gate("X", targets=1)
    return circuit


def _circuit_cnot_z():
    circuit = QubitCircuit(2)
    circuit.add_gate("Z", targets=0)
    circuit.add_gate("CNOT", targets=0, controls=1)
    circuit.add_gate("Z", targets=1)
    return circuit


@pytest.mark.parametrize(
    "circuit, expected_length",
    [
        pytest.param(_circuit_cnot_x(), 2, id="cnot x commutation"),
        pytest.param(_circuit_cnot_z(), 2, id="cnot z commutation"),
    ],
)
def test_commutation_rules(circuit, expected_length):
    scheduler = Scheduler("ASAP")
    gate_cycle_indices = scheduler.schedule(circuit)
    assert (max(gate_cycle_indices) + 1) == expected_length
    assert _verify_scheduled_circuit(circuit, gate_cycle_indices)


def _circuit1():
    circuit1 = QubitCircuit(6)
    circuit1.add_gate("SNOT", targets=2)
    circuit1.add_gate("CNOT", targets=4, controls=2)
    circuit1.add_gate("CNOT", targets=3, controls=2)
    circuit1.add_gate("CNOT", targets=1, controls=2)
    circuit1.add_gate("CNOT", targets=5, controls=4)
    circuit1.add_gate("CNOT", targets=1, controls=5)
    circuit1.add_gate("SWAP", targets=[0, 1])
    return circuit1


def _circuit2():
    circuit2 = QubitCircuit(8)
    circuit2.add_gate("SNOT", targets=1)
    circuit2.add_gate("SNOT", targets=2)
    circuit2.add_gate("SNOT", targets=3)
    circuit2.add_gate("CNOT", targets=4, controls=5)
    circuit2.add_gate("CNOT", targets=4, controls=6)
    circuit2.add_gate("CNOT", targets=3, controls=4)
    circuit2.add_gate("CNOT", targets=3, controls=5)
    circuit2.add_gate("CNOT", targets=3, controls=7)
    circuit2.add_gate("CNOT", targets=2, controls=4)
    circuit2.add_gate("CNOT", targets=2, controls=6)
    circuit2.add_gate("CNOT", targets=2, controls=7)
    circuit2.add_gate("CNOT", targets=1, controls=5)
    circuit2.add_gate("CNOT", targets=1, controls=6)
    circuit2.add_gate("CNOT", targets=1, controls=7)
    return circuit2


def _instructions1():
    circuit3 = QubitCircuit(6)
    circuit3.add_gate("SNOT", targets=0)
    circuit3.add_gate("SNOT", targets=1)
    circuit3.add_gate("CNOT", targets=2, controls=3)
    circuit3.add_gate("CNOT", targets=1, controls=2)
    circuit3.add_gate("CNOT", targets=1, controls=0)
    circuit3.add_gate("SNOT", targets=3)
    circuit3.add_gate("CNOT", targets=1, controls=3)
    circuit3.add_gate("CNOT", targets=1, controls=3)

    instruction_list = []
    for circuit_ins in circuit3.instructions:
        if circuit_ins.operation.name == "SNOT":
            instruction_list.append(Instruction(circuit_ins, duration=1))
        else:
            instruction_list.append(Instruction(circuit_ins, duration=2))

    return instruction_list


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(
            deepcopy(_circuit1()), "ASAP", 4, False, False, id="circuit1 ASAP)"
        ),
        pytest.param(
            deepcopy(_circuit1()), "ALAP", 4, False, False, id="circuit1 ALAP)"
        ),
    ],
)
def test_scheduling_gates1(
    circuit, method, expected_length, random_shuffle, gates_schedule
):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num
    )

    assert max(gate_cycle_indices) == expected_length
    _verify_scheduled_circuit(circuit, gate_cycle_indices)


# There is some problem with circuit2 on Mac.
# When I list all three tests together, a segment fault arise.
# I suspect it has something to do with pytest.
@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(
            deepcopy(_circuit2()), "ASAP", 4, False, False, id="circuit2 ASAP"
        ),
    ],
)
def test_scheduling_gates2(
    circuit, method, expected_length, random_shuffle, gates_schedule
):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num
    )

    assert max(gate_cycle_indices) == expected_length
    _verify_scheduled_circuit(circuit, gate_cycle_indices)


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(
            deepcopy(_circuit2()),
            "ALAP",
            5,
            False,
            False,
            id="circuit2 ALAP no shuffle",
        )
    ],
)
def test_scheduling_gates3(
    circuit, method, expected_length, random_shuffle, gates_schedule
):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num
    )

    assert max(gate_cycle_indices) == expected_length
    _verify_scheduled_circuit(circuit, gate_cycle_indices)


@pytest.mark.parametrize(
    "circuit, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(
            deepcopy(_circuit2()),
            "ALAP",
            4,
            True,
            False,
            id="circuit2 ALAP shuffle",
        ),  # with random shuffling
    ],
)
def test_scheduling_gates4(
    circuit, method, expected_length, random_shuffle, gates_schedule
):
    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        circuit, gates_schedule=gates_schedule, repeat_num=repeat_num
    )

    assert max(gate_cycle_indices) == expected_length
    _verify_scheduled_circuit(circuit, gate_cycle_indices)


@pytest.mark.parametrize(
    "instructions, method, expected_length, random_shuffle, gates_schedule",
    [
        pytest.param(
            deepcopy(_instructions1()),
            "ASAP",
            7,
            False,
            False,
            id="circuit1 pulse ASAP)",
        ),
        pytest.param(
            deepcopy(_instructions1()),
            "ALAP",
            7,
            False,
            False,
            id="circuit1 pulse ALAP)",
        ),
    ],
)
def test_scheduling_pulse(
    instructions, method, expected_length, random_shuffle, gates_schedule
):
    circuit = QubitCircuit(4)
    for instruction in instructions:
        gate_cls = GATE_CLASS_MAP[instruction.name]

        if issubclass(gate_cls, ControlledGate):
            circuit.add_gate(
                gate_cls,
                targets=instruction.targets,
                controls=instruction.controls,
            )
        else:
            circuit.add_gate(gate_cls, targets=instruction.targets)

    if random_shuffle:
        repeat_num = 5
    else:
        repeat_num = 0

    # run the scheduler
    scheduler = Scheduler(method)
    gate_cycle_indices = scheduler.schedule(
        instructions, gates_schedule=gates_schedule, repeat_num=repeat_num
    )
    assert max(gate_cycle_indices) == expected_length
