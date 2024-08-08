import numpy as np
import pytest
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, Measurement
from qutip_qip.circuit.text_renderer import TextRenderer


@pytest.fixture
def qc1():
    """
    Fixture to create and provide a QubitCircuit with predefined gates.
    """
    qc = QubitCircuit(4)
    qc.add_gate("ISWAP", targets=[0, 1])
    qc.add_gate("ISWAP", targets=[0, 1])
    qc.add_gate("CTRLRX", targets=[0, 1], controls=[2, 3], arg_value=np.pi / 2)
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_gate("BERKELEY", targets=[0, 3])
    qc.add_gate("FREDKIN", controls=[3], targets=[1, 2])
    qc.add_gate("CX", controls=[0], targets=[1])
    qc.add_gate("CRX", controls=[2], targets=[3], arg_value=0.5)
    qc.add_gate("SWAP", targets=[0, 3])
    return qc


@pytest.fixture
def qc2():
    """
    Fixture to create and provide a QubitCircuit with predefined gates.
    """
    qc = QubitCircuit(4, num_cbits=2)
    qc.add_gate("H", targets=[0])
    qc.add_gate("H", targets=[0])
    qc.add_gate("CNOT", controls=[1], targets=[0])
    qc.add_gate("X", targets=[2])
    qc.add_gate("CNOT", controls=[0], targets=[1])
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_gate("BERKELEY", targets=[0, 3])
    qc.add_gate("FREDKIN", controls=[3], targets=[1, 2])
    qc.add_gate("CX", controls=[0], targets=[1])
    qc.add_gate("CRX", controls=[0], targets=[1], arg_value=0.5)
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_measurement("M", targets=[0], classical_store=0)
    qc.add_measurement("M", targets=[1], classical_store=1)
    return qc


@pytest.mark.parametrize("qc_fixture", ["qc1", "qc2"])
def test_parts_len(request, qc_fixture):
    """
    Check if the length of different parts of the gate is the same.
    """
    qc = request.getfixturevalue(qc_fixture)
    tr = TextRenderer(qc)
    for gate in qc.gates:
        if isinstance(gate, Gate):
            if len(gate.targets) == 1 and gate.controls is None:
                parts, _ = tr._draw_singleq_gate(gate.name)
            else:
                parts, _ = tr._draw_multiq_gate(gate)
        elif isinstance(gate, Measurement):
            parts, _ = tr._draw_measurement_gate(gate)

        len_parts = [len(part) for part in parts]
        assert (
            len(set(len_parts)) == 1
        ), f"Gate {gate} has parts with different lengths: {len_parts}"


@pytest.mark.parametrize("qc_fixture", ["qc1", "qc2"])
def test_render_str_len(request, qc_fixture):
    """
    Check if all render wire lengths are the same.
    """
    qc = request.getfixturevalue(qc_fixture)
    tr = TextRenderer(qc)
    tr.layout()
    render_str = tr._render_strs

    assert (
        len(set([len(wire) for wire in render_str])) == 1
    ), "Render wires have different lengths."
