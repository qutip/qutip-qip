import pytest
import numpy as np
from unittest.mock import patch
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit.draw import TeXRenderer, TextRenderer
from qutip_qip.circuit.draw.base_renderer import format_gate_parameters
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
    R,
    RZ,
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


def _text_mid(qc):
    renderer = TextRenderer(qc)
    renderer.layout()
    return "\n".join(renderer._render_strs["mid_frame"])


def test_format_gate_parameters():
    assert format_gate_parameters(np.pi / 2) == "π/2"
    assert format_gate_parameters(-np.pi / 2) == "-π/2"
    assert format_gate_parameters(np.pi) == "π"
    assert format_gate_parameters(-3 * np.pi / 4) == "-3π/4"
    assert format_gate_parameters(2 * np.pi) == "2π"
    assert format_gate_parameters(0) == "0"
    assert format_gate_parameters(0.5) == "0.50"
    assert format_gate_parameters((np.pi / 2, 0.5)) == "π/2, 0.50"
    assert format_gate_parameters(np.pi / 2, latex=True) == r"\pi/2"
    assert format_gate_parameters((-np.pi, np.pi / 4), latex=True) == r"-\pi, \pi/4"


def test_showarg_text_renderer():
    shown = QubitCircuit(1)
    shown.add_gate(RZ(np.pi / 2), targets=0, style={"showarg": True})
    assert "RZ(π/2)" in _text_mid(shown)

    hidden = QubitCircuit(1)
    hidden.add_gate(RZ(np.pi / 2), targets=0, style={"showarg": False})
    hidden_mid = _text_mid(hidden)
    assert "RZ" in hidden_mid
    assert "π/2" not in hidden_mid

    numeric = QubitCircuit(1)
    numeric.add_gate(RZ(0.5), targets=0, style={"showarg": True})
    assert "RZ(0.50)" in _text_mid(numeric)

    multi = QubitCircuit(1)
    multi.add_gate(R((np.pi / 2, np.pi / 4)), targets=0, style={"showarg": True})
    assert "R(π/2, π/4)" in _text_mid(multi)

    controlled = QubitCircuit(2)
    controlled.add_gate(CRX(np.pi / 2), targets=0, controls=1, style={"showarg": True})
    assert "CRX(π/2)" in _text_mid(controlled)

    plain = QubitCircuit(1)
    plain.add_gate(X, targets=0, style={"showarg": True})
    assert "RZ" not in _text_mid(plain)
    assert "┤ X ├" in _text_mid(plain)


def test_showarg_does_not_replace_arg_label():
    symbolic = QubitCircuit(1)
    symbolic.add_gate(RZ(np.pi / 2, arg_label=r"\phi"), targets=0)
    symbolic_mid = _text_mid(symbolic)
    assert r"\phi" in symbolic_mid
    assert "π/2" not in symbolic_mid

    both = QubitCircuit(1)
    both.add_gate(
        RZ(np.pi / 2, arg_label=r"\phi"),
        targets=0,
        style={"showarg": True},
    )
    assert "RZ(π/2)" in _text_mid(both)


def test_showarg_deprecated_add_gate_api():
    qc = QubitCircuit(1)
    with pytest.warns(DeprecationWarning):
        qc.add_gate("RZ", targets=0, arg_value=np.pi / 2, style={"showarg": True})
    assert "RZ(π/2)" in _text_mid(qc)


def test_showarg_latex_renderer():
    shown = QubitCircuit(1)
    shown.add_gate(RZ(np.pi / 2), targets=0, style={"showarg": True})
    latex = TeXRenderer(shown).latex_code()
    assert r"\gate{R_z(\pi/2)}" in latex

    hidden = QubitCircuit(1)
    hidden.add_gate(RZ(np.pi / 2), targets=0)
    hidden_latex = TeXRenderer(hidden).latex_code()
    assert r"\gate{R_z}" in hidden_latex
    assert r"\pi" not in hidden_latex

    symbolic = QubitCircuit(1)
    symbolic.add_gate(RZ(np.pi / 2, arg_label=r"\phi"), targets=0)
    assert r"\gate{R_z(\phi)}" in TeXRenderer(symbolic).latex_code()

    controlled = QubitCircuit(2)
    controlled.add_gate(CRX(np.pi / 2), targets=1, controls=0, style={"showarg": True})
    assert r"\gate{{\rm CRX}(\pi/2)}" in TeXRenderer(controlled).latex_code()

    multi = QubitCircuit(1)
    multi.add_gate(R((np.pi / 2, 0.2)), targets=0, style={"showarg": True})
    assert r"\gate{{\rm R}(\pi/2, 0.20)}" in TeXRenderer(multi).latex_code()


def test_showarg_matplotlib_renderer():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qutip_qip.circuit.draw import MatRenderer

    shown = QubitCircuit(1)
    shown.add_gate(RZ(np.pi / 2), targets=0, style={"showarg": True})
    with patch("matplotlib.pyplot.show"):
        renderer = MatRenderer(shown)
        renderer.canvas_plot()
    try:
        labels = [text.get_text() for text in renderer._ax.texts]
    finally:
        plt.close(renderer.fig)
    assert "RZ(π/2)" in labels

    hidden = QubitCircuit(2)
    hidden.add_gate(CRX(0.5), targets=1, controls=0, style={"showarg": False})
    with patch("matplotlib.pyplot.show"):
        renderer = MatRenderer(hidden)
        renderer.canvas_plot()
    try:
        labels = [text.get_text() for text in renderer._ax.texts]
    finally:
        plt.close(renderer.fig)
    assert "CRX" in labels
    assert "CRX(0.50)" not in labels
