import pytest
import numpy as np
from unittest.mock import patch
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit.draw import TextRenderer
from qutip_qip.operations import get_controlled_gate
from qutip_qip.operations.gates import (
    IDENTITY,
    X,
    H,
    CX,
    CRX,
    CPHASE,
    SWAP,
    ISWAP,
    TOFFOLI,
    FREDKIN,
    BERKELEY,
)
from qutip_qip.operations.measurement import Mz


@pytest.fixture
def qc1():
    qc = QubitCircuit(4)
    qc.add_gate(ISWAP, targets=[2, 3])
    qc.add_gate(CRX(np.pi / 2), targets=[0], controls=[1])
    qc.add_gate(SWAP, targets=[0, 3])
    qc.add_gate(BERKELEY, targets=[0, 3])
    qc.add_gate(FREDKIN, controls=[3], targets=[1, 2])
    qc.add_gate(TOFFOLI, controls=[0, 2], targets=[1])
    qc.add_gate(CX, controls=[0], targets=[1])
    qc.add_gate(CRX(0.5), controls=[2], targets=[3])
    qc.add_gate(SWAP, targets=[0, 3])
    return qc


def test_layout_qc1(qc1):
    tr = TextRenderer(qc1)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "        ┌──┴──┐     │   │          │                    │          │      │    ",
            "                    │   │          │  │         │  ┌────┴────┐  ┌────┐    │    ",
            "        │       │   │   │          │  ┌────┴────┐                  │      │    ",
            "        ┌───────┐       ┌──────────┐                            ┌─────┐        ",
        ],
        "mid_frame": [
            " q0 :───┤ CRX ├─────╳───┤ BERKELEY ├────────────────────█──────────█──────╳────",
            " q1 :──────█────────│── │          │ ─┤ FREDKIN ├──┤ TOFFOLI ├──┤ CX ├────│────",
            " q2 :───┤ ISWAP ├───│── │          │ ─┤         ├───────█──────────█──────│────",
            " q3 :───┤       ├───╳───┤          ├───────█────────────────────┤ CRX ├───╳────",
        ],
        "bot_frame": [
            "        └─────┘         └──────────┘                                           ",
            "           │        │   │          │  └─────────┘  └────┬────┘  └──┬─┘    │    ",
            "        └───────┘   │   │          │  │         │       │                 │    ",
            "        │       │   │   │          │       │                    └──┬──┘   │    ",
        ],
    }


@pytest.fixture
def qc2():
    qc = QubitCircuit(4, num_cbits=2)
    qc.add_gate(H, targets=[0])
    qc.add_gate(H, targets=[0])
    qc.add_gate(CX, controls=[1], targets=[0])
    qc.add_gate(X, targets=[2])
    qc.add_gate(CX, controls=[0], targets=[1])
    qc.add_gate(SWAP, targets=[0, 3])
    qc.add_gate(BERKELEY, targets=[0, 3])
    qc.add_gate(FREDKIN, controls=[3], targets=[1, 2])
    qc.add_gate(CX, controls=[0], targets=[1])
    qc.add_gate(CRX(0.5), controls=[0], targets=[1])
    qc.add_gate(SWAP, targets=[0, 3])
    qc.add_gate(SWAP, targets=[0, 3])
    qc.add_measurement(Mz, targets=[0], classical_store=0)
    qc.add_measurement(Mz, targets=[1], classical_store=1)
    return qc


def test_layout_qc2(qc2):
    tr = TextRenderer(qc2)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "        ┌───┐  ┌───┐  ┌──┴─┐     │     │   │          │                  │       │      │    │   ┌───┐    ║     ",
            "                              ┌────┐   │   │          │  │         │  ┌────┐  ┌─────┐   │    │          ┌───┐   ",
            "        ┌───┐                          │   │          │  ┌────┴────┐                    │    │                  ",
            "                                           ┌──────────┐                                                         ",
            "                                                                                                   ║            ",
            "                                                                                                   ║      ║     ",
        ],
        "mid_frame": [
            " q0 :───┤ H ├──┤ H ├──┤ CX ├─────█─────╳───┤ BERKELEY ├──────────────────█───────█──────╳────╳───┤ M ├────║─────",
            " q1 :────────────────────█────┤ CX ├───│── │          │ ─┤ FREDKIN ├──┤ CX ├──┤ CRX ├───│────│──────────┤ M ├───",
            " q2 :───┤ X ├──────────────────────────│── │          │ ─┤         ├────────────────────│────│──────────────────",
            " q3 :──────────────────────────────────╳───┤          ├───────█─────────────────────────╳────╳──────────────────",
            " c0 :══════════════════════════════════════════════════════════════════════════════════════════════╩════════════",
            " c1 :══════════════════════════════════════════════════════════════════════════════════════════════║══════╩═════",
        ],
        "bot_frame": [
            "        └───┘  └───┘  └────┘               └──────────┘                                          └─╥─┘    ║     ",
            "                         │    └──┬─┘   │   │          │  └─────────┘  └──┬─┘  └──┬──┘   │    │          └─╥─┘   ",
            "        └───┘                          │   │          │  │         │                    │    │                  ",
            "                                       │   │          │       │                         │    │                  ",
            "                                                                                                                ",
            "                                                                                                   ║            ",
        ],
    }


def test_layout_qc3(qc3):
    tr = TextRenderer(qc3)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "        ┌───┐  ┌──┴─┐     │     │   │          │                  │       │     ┌───┐   │        │                                ║     ",
            "                       ┌────┐   │   │          │  │         │  ┌────┐  ┌─────┐          │        │                   │       │  ┌───┐   ",
            "        ┌───┐                   │   │          │  ┌────┴────┐                           │   ┌─────────┐       │      │       │          ",
            "                                    ┌──────────┐                                                         ┌────────┐  ┌───────┐          ",
            "                                                                                  ║                                                     ",
            "                                                                                  ║                                               ║     ",
        ],
        "mid_frame": [
            " q0 :───┤ H ├──┤ CX ├─────█─────╳───┤ BERKELEY ├──────────────────█───────█─────┤ M ├───╳────────█────────────────────────────────║─────",
            " q1 :─────────────█────┤ CX ├───│── │          │ ─┤ FREDKIN ├──┤ CX ├──┤ CRX ├──────────│────────█───────────────────┤ ISWAP ├──┤ M ├───",
            " q2 :───┤ X ├───────────────────│── │          │ ─┤         ├───────────────────────────│───┤ TOFFOLI ├───────█───── │       │ ─────────",
            " q3 :───────────────────────────╳───┤          ├───────█────────────────────────────────╳────────────────┤ CPHASE ├──┤       ├──────────",
            " c0 :═════════════════════════════════════════════════════════════════════════════╩═════════════════════════════════════════════════════",
            " c1 :═════════════════════════════════════════════════════════════════════════════║═══════════════════════════════════════════════╩═════",
        ],
        "bot_frame": [
            "        └───┘  └────┘               └──────────┘                                └─╥─┘                                             ║     ",
            "                  │    └──┬─┘   │   │          │  └─────────┘  └──┬─┘  └──┬──┘          │        │                   └───────┘  └─╥─┘   ",
            "        └───┘                   │   │          │  │         │                           │   └────┬────┘              │       │          ",
            "                                │   │          │       │                                │                └────┬───┘  │       │          ",
            "                                                                                                                                        ",
            "                                                                                  ║                                                     ",
        ],
    }


@pytest.fixture
def qc3():
    qc = QubitCircuit(4, num_cbits=2)
    qc.add_gate(H, targets=[0])
    qc.add_gate(CX, controls=[1], targets=[0])
    qc.add_gate(X, targets=[2])
    qc.add_gate(CX, controls=[0], targets=[1])
    qc.add_gate(SWAP, targets=[0, 3])
    qc.add_gate(BERKELEY, targets=[0, 3])
    qc.add_gate(FREDKIN, controls=[3], targets=[1, 2])
    qc.add_gate(CX, controls=[0], targets=[1])
    qc.add_gate(CRX(0.5), controls=[0], targets=[1])
    qc.add_measurement(Mz, targets=[0], classical_store=0)
    qc.add_gate(SWAP, targets=[0, 3])
    qc.add_gate(TOFFOLI, controls=[0, 1], targets=[2])
    qc.add_gate(CPHASE(0.75), controls=[2], targets=[3])
    qc.add_gate(ISWAP, targets=[1, 3])
    qc.add_measurement(Mz, targets=[1], classical_store=1)
    return qc


@pytest.fixture
def qc4():
    i = get_controlled_gate(IDENTITY, n_ctrl_qubits=1, gate_name="i")
    ii = get_controlled_gate(IDENTITY, n_ctrl_qubits=2, gate_name="ii")
    iii = get_controlled_gate(
        IDENTITY, n_ctrl_qubits=1, control_value=0, gate_name="iii"
    )

    qc = QubitCircuit(5, num_cbits=2)
    qc.add_gate(X, targets=0, classical_controls=[0, 1], classical_control_value=0)
    qc.add_gate(i, targets=1, controls=2)
    qc.add_gate(
        ii,
        targets=1,
        classical_controls=1,
        controls=[3, 4],
        classical_control_value=1,
    )
    qc.add_gate(
        iii,
        targets=1,
        classical_controls=1,
        controls=4,
        classical_control_value=1,
    )
    qc.add_gate(ii, targets=2, controls=[4, 3])
    qc.add_gate(SWAP, targets=[0, 1])
    return qc


def test_layout_qc4(qc4):
    tr = TextRenderer(qc4)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "        ┌───┐     ║       ║      │       ",
            "        ┌─┴─┐  ┌──┴─┐  ┌──┴──┐           ",
            "                  │       │     ┌──┴─┐   ",
            "                  │       │        │     ",
            "                                         ",
            "          ║                              ",
            "          ║       ║       ║              ",
        ],
        "mid_frame": [
            " q0 :───┤ X ├─────║───────║──────╳───────",
            " q1 :───┤ i ├──┤ ii ├──┤ iii ├───╳───────",
            " q2 :─────█───────│───────│─────┤ ii ├───",
            " q3 :─────────────█───────│────────█─────",
            " q4 :─────────────█───────█────────█─────",
            " c0 :═════█══════════════════════════════",
            " c1 :═════█═══════█═══════█══════════════",
        ],
        "bot_frame": [
            "        └─╥─┘     ║       ║              ",
            "        └───┘  └──╥─┘  └──╥──┘   │       ",
            "          │       │       │     └────┘   ",
            "                  │       │        │     ",
            "                  │       │        │     ",
            "                                         ",
            "          ║                              ",
        ],
    }


@pytest.mark.parametrize("qc_fixture", ["qc1", "qc2", "qc3"])
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


@pytest.mark.parametrize("qc_fixture", ["qc1", "qc2", "qc3"])
def test_matrenderer(request, qc_fixture):
    """
    Check if Matplotlib renderer works without error.
    """
    pytest.importorskip("matplotlib")
    qc = request.getfixturevalue(qc_fixture)

    with patch("matplotlib.pyplot.show"):  # to avoid showing the plot
        qc.draw("matplotlib")


@pytest.mark.parametrize("qc_fixture", ["qc1", "qc2", "qc3"])
def test_circuit_saving(request, qc_fixture, tmpdir):
    """
    Test if the different renderers can save the circuit in different formats.
    """
    pytest.importorskip("matplotlib")
    qc = request.getfixturevalue(qc_fixture)

    # test MatRenderer
    with patch("matplotlib.pyplot.show"):  # to avoid showing the plot
        qc.draw("matplotlib", save=True, file_path=str(tmpdir.join("test")))
    assert tmpdir.join("test.png").check(), "MatRenderer saved PNG file not found."

    # test TextRenderer
    qc.draw("text", save=True, file_path=str(tmpdir.join("test")))
    assert tmpdir.join("test.txt").check(), "TextRenderer saved TXT file not found."


def _control_nodes(qc):
    """
    Render ``qc`` and return the drawn control nodes with the wire separation.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    from qutip_qip.circuit.draw import MatRenderer

    # Draw on an Agg canvas so the test does not depend on a GUI backend.
    fig = Figure()
    FigureCanvasAgg(fig)
    with patch("matplotlib.pyplot.show"):  # to avoid showing the plot
        renderer = MatRenderer(qc, ax=fig.add_subplot(111))
        renderer.canvas_plot()

    control_nodes = [
        drawn
        for drawn in renderer._ax.patches
        if isinstance(drawn, Circle) and drawn.radius == renderer._control_node_r
    ]
    return control_nodes, renderer.style.wire_sep


def _is_filled(control_node) -> bool:
    """A control node is filled when its face and edge share the gate color."""
    return tuple(control_node.get_facecolor()) == tuple(control_node.get_edgecolor())


@pytest.mark.parametrize("control_value, filled", [(1, True), (0, False)])
def test_matrenderer_control_value_sets_the_control_fill(control_value, filled):
    """
    Check that a control activated by |1> is filled and one by |0> is hollow.
    """
    pytest.importorskip("matplotlib")
    controlled_x = get_controlled_gate(X, n_ctrl_qubits=1, control_value=control_value)
    qc = QubitCircuit(2)
    qc.add_gate(controlled_x, controls=[0], targets=[1])

    control_nodes, _ = _control_nodes(qc)

    assert len(control_nodes) == 1
    assert _is_filled(control_nodes[0]) is filled


def test_matrenderer_control_values_follow_the_ctrl_value_bits():
    """
    Check that each control is filled according to the gate's own semantics.

    Every control qubit is drawn from the matching bit of ``ctrl_value``, the
    most significant bit belonging to the first control qubit. For
    ``ctrl_value=0b01`` with two controls the gate applies when the first
    control is |0> and the second is |1>, so the hollow control must be drawn
    on the first control qubit.
    """
    pytest.importorskip("matplotlib")
    controlled_x = get_controlled_gate(X, n_ctrl_qubits=2, control_value=0b01)
    qc = QubitCircuit(3)
    qc.add_gate(controlled_x, controls=[0, 1], targets=[2])

    # The unitary confirms the mapping the rendering has to follow.
    unitary = qc.compute_unitary().full()
    flipped_states = {
        format(index, "03b") for index in range(8) if abs(unitary[index, index]) < 0.5
    }
    assert flipped_states == {"010", "011"}

    control_nodes, wire_sep = _control_nodes(qc)
    assert len(control_nodes) == 2

    fills = {
        round(control_node.center[1] / wire_sep): _is_filled(control_node)
        for control_node in control_nodes
    }
    assert fills == {0: False, 1: True}
