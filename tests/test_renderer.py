import pytest
import numpy as np
from unittest.mock import patch
from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit.text_renderer import TextRenderer


@pytest.fixture
def qc1():
    qc = QubitCircuit(4)
    qc.add_gate("ISWAP", targets=[2, 3])
    qc.add_gate("CTRLRX", targets=[0, 1], controls=[2, 3], arg_value=np.pi / 2)
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_gate("BERKELEY", targets=[0, 3])
    qc.add_gate("FREDKIN", controls=[3], targets=[1, 2])
    qc.add_gate("TOFFOLI", controls=[0, 2], targets=[1])
    qc.add_gate("CX", controls=[0], targets=[1])
    qc.add_gate("CRX", controls=[2], targets=[3], arg_value=0.5)
    qc.add_gate("SWAP", targets=[0, 3])
    return qc


@pytest.fixture
def qc2():
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


@pytest.fixture
def qc3():
    qc = QubitCircuit(4, num_cbits=2)
    qc.add_gate("H", targets=[0])
    qc.add_gate("CNOT", controls=[1], targets=[0])
    qc.add_gate("X", targets=[2])
    qc.add_gate("CNOT", controls=[0], targets=[1])
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_gate("BERKELEY", targets=[0, 3])
    qc.add_gate("FREDKIN", controls=[3], targets=[1, 2])
    qc.add_gate("CX", controls=[0], targets=[1])
    qc.add_gate("CRX", controls=[0], targets=[1], arg_value=0.5)
    qc.add_measurement("M", targets=[0], classical_store=0)
    qc.add_gate("SWAP", targets=[0, 3])
    qc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
    qc.add_gate("CPHASE", controls=[2], targets=[3], arg_value=0.75)
    qc.add_gate("ISWAP", targets=[1, 3])
    qc.add_measurement("M", targets=[1], classical_store=1)
    return qc


def test_layout_qc1(qc1):
    tr = TextRenderer(qc1)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "                   │        │   │   │          │                    │          │      │    ",
            "                   ┌────┴───┐   │   │          │  │         │  ┌────┴────┐  ┌────┐    │    ",
            "        │       │       │       │   │          │  ┌────┴────┐                  │      │    ",
            "        ┌───────┐                   ┌──────────┐                            ┌─────┐        ",
        ],
        "mid_frame": [
            " q0 :──────────────┤ CTRLRX ├───╳───┤ BERKELEY ├────────────────────█──────────█──────╳────",
            " q1 :──────────────┤        ├───│── │          │ ─┤ FREDKIN ├──┤ TOFFOLI ├──┤ CX ├────│────",
            " q2 :───┤ ISWAP ├───────█───────│── │          │ ─┤         ├───────█──────────█──────│────",
            " q3 :───┤       ├───────█───────╳───┤          ├───────█────────────────────┤ CRX ├───╳────",
        ],
        "bot_frame": [
            "                   └────────┘       └──────────┘                                           ",
            "                   │        │   │   │          │  └─────────┘  └────┬────┘  └──┬─┘    │    ",
            "        └───────┘       │       │   │          │  │         │       │                 │    ",
            "        │       │       │       │   │          │       │                    └──┬──┘   │    ",
        ],
    }


def test_layout_qc2(qc2):
    tr = TextRenderer(qc2)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "        ┌───┐  ┌───┐  ┌───┴──┐      │      │   │          │                  │       │      │    │   ┌───┐    ║     ",
            "                                ┌──────┐   │   │          │  │         │  ┌────┐  ┌─────┐   │    │          ┌───┐   ",
            "        ┌───┐                              │   │          │  ┌────┴────┐                    │    │                  ",
            "                                               ┌──────────┐                                                         ",
            "                                                                                                       ║            ",
            "                                                                                                       ║      ║     ",
        ],
        "mid_frame": [
            " q0 :───┤ H ├──┤ H ├──┤ CNOT ├──────█──────╳───┤ BERKELEY ├──────────────────█───────█──────╳────╳───┤ M ├────║─────",
            " q1 :─────────────────────█─────┤ CNOT ├───│── │          │ ─┤ FREDKIN ├──┤ CX ├──┤ CRX ├───│────│──────────┤ M ├───",
            " q2 :───┤ X ├──────────────────────────────│── │          │ ─┤         ├────────────────────│────│──────────────────",
            " q3 :──────────────────────────────────────╳───┤          ├───────█─────────────────────────╳────╳──────────────────",
            " c0 :══════════════════════════════════════════════════════════════════════════════════════════════════╩════════════",
            " c1 :══════════════════════════════════════════════════════════════════════════════════════════════════║══════╩═════",
        ],
        "bot_frame": [
            "        └───┘  └───┘  └──────┘                 └──────────┘                                          └─╥─┘    ║     ",
            "                          │     └───┬──┘   │   │          │  └─────────┘  └──┬─┘  └──┬──┘   │    │          └─╥─┘   ",
            "        └───┘                              │   │          │  │         │                    │    │                  ",
            "                                           │   │          │       │                         │    │                  ",
            "                                                                                                                    ",
            "                                                                                                       ║            ",
        ],
    }


def test_layout_qc3(qc3):
    tr = TextRenderer(qc3)
    tr.layout()
    assert tr._render_strs == {
        "top_frame": [
            "        ┌───┐  ┌───┴──┐      │      │   │          │                  │       │     ┌───┐   │        │                                ║     ",
            "                         ┌──────┐   │   │          │  │         │  ┌────┐  ┌─────┐          │        │                   │       │  ┌───┐   ",
            "        ┌───┐                       │   │          │  ┌────┴────┐                           │   ┌─────────┐       │      │       │          ",
            "                                        ┌──────────┐                                                         ┌────────┐  ┌───────┐          ",
            "                                                                                      ║                                                     ",
            "                                                                                      ║                                               ║     ",
        ],
        "mid_frame": [
            " q0 :───┤ H ├──┤ CNOT ├──────█──────╳───┤ BERKELEY ├──────────────────█───────█─────┤ M ├───╳────────█────────────────────────────────║─────",
            " q1 :──────────────█─────┤ CNOT ├───│── │          │ ─┤ FREDKIN ├──┤ CX ├──┤ CRX ├──────────│────────█───────────────────┤ ISWAP ├──┤ M ├───",
            " q2 :───┤ X ├───────────────────────│── │          │ ─┤         ├───────────────────────────│───┤ TOFFOLI ├───────█───── │       │ ─────────",
            " q3 :───────────────────────────────╳───┤          ├───────█────────────────────────────────╳────────────────┤ CPHASE ├──┤       ├──────────",
            " c0 :═════════════════════════════════════════════════════════════════════════════════╩═════════════════════════════════════════════════════",
            " c1 :═════════════════════════════════════════════════════════════════════════════════║═══════════════════════════════════════════════╩═════",
        ],
        "bot_frame": [
            "        └───┘  └──────┘                 └──────────┘                                └─╥─┘                                             ║     ",
            "                   │     └───┬──┘   │   │          │  └─────────┘  └──┬─┘  └──┬──┘          │        │                   └───────┘  └─╥─┘   ",
            "        └───┘                       │   │          │  │         │                           │   └────┬────┘              │       │          ",
            "                                    │   │          │       │                                │                └────┬───┘  │       │          ",
            "                                                                                                                                            ",
            "                                                                                      ║                                                     ",
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
    qc = request.getfixturevalue(qc_fixture)

    with patch("matplotlib.pyplot.show"):
        qc.draw("matplotlib")
