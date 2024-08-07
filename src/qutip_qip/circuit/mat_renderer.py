"""
Module for rendering a quantum circuit using matplotlib library.
"""

from typing import Union, Optional, List, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import (
    FancyBboxPatch,
    Circle,
    Arc,
    FancyArrow,
)

from ..operations import Gate, Measurement
from ..circuit import QubitCircuit
from .color_theme import qutip, light, dark, modern

__all__ = [
    "MatRenderer",
]


@dataclass
class StyleConfig:
    """
    Dataclass to store the style configuration for circuit customization.

    Parameters
    ----------
    dpi : int, optional
        The dpi of the figure. The default is 150.

    fontsize : int, optional
        The fontsize control at circuit level, including tile and wire labels. The default is 10.

    end_wire_ext : int, optional
        The extension of the wire at the end of the circuit. The default is 2.

    padding : float, optional
        The padding between the circuit and the figure border. The default is 0.3.

    gate_margin : float, optional
        The margin space left on each side of the gate. The default is 0.15.

    wire_sep : float, optional
        The separation between the wires. The default is 0.5.

    layer_sep : float, optional
        The separation between the layers. The default is 0.5.

    gate_pad : float, optional
        The padding between the gate and the gate label. The default is 0.05.

    label_pad : float, optional
        The padding between the wire label and the wire. The default is 0.1.

    fig_height : float, optional
        The height of the figure. The default is None.

    fig_width : float, optional
        The width of the figure. The default is None.

    bulge : Union[str, bool], optional
        The bulge style of the gate. Renders non-bulge gates if False. The default is True.

    align_layer : bool, optional
        Align the layers of the gates across different wires. The default is False.

    theme : Optional[Union[str, Dict]], optional
        The color theme of the circuit. The default is "qutip".
        The available themes are 'qutip', 'light', and 'modern'.

    title : Optional[str], optional
        The title of the circuit. The default is None.

    bgcolor : Optional[str], optional
        The background color of the circuit. The default is None.

    color : Optional[str], optional
        Controls color of acsent elements (eg. cross sign in the target node)
        and set as deafult color of gate-label. Can be overwritten by gate specific color.
        The default is None.

    wire_label : Optional[List], optional
        The labels of the wires. The default is None.

    wire_color : Optional[str], optional
        The color of the wires. The default is None.
    """

    dpi: int = 150
    fontsize: int = 10
    end_wire_ext: int = 2
    padding: float = 0.3
    gate_margin: float = 0.15
    wire_sep: float = 0.5
    layer_sep: float = 0.5
    gate_pad: float = 0.05
    label_pad: float = 0.1
    fig_height: Optional[float] = None
    fig_width: Optional[float] = 10
    bulge: Union[str, bool] = True
    align_layer: bool = False
    theme: Optional[Union[str, Dict]] = "qutip"
    title: Optional[str] = None
    bgcolor: Optional[str] = None
    color: Optional[str] = None
    wire_label: Optional[List] = None
    wire_color: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.bulge, bool):
            self.bulge = "round4" if self.bulge else "square"

        self.measure_color = "#000000"
        if self.theme == "qutip":
            self.theme = qutip
        elif self.theme == "light":
            self.theme = light
        elif self.theme == "dark":
            self.theme = dark
            self.measure_color = "#FFFFFF"
        elif self.theme == "modern":
            self.theme = modern
        else:
            raise ValueError(
                f"""Invalid theme: {self.theme},
                Must be selectec from 'qutip', 'light', 'dark', or 'modern'.
                """
            )

        self.bgcolor = self.bgcolor or self.theme["bgcolor"]
        self.color = self.color or self.theme["color"]
        self.wire_color = self.wire_color or self.theme["wire_color"]


class MatRenderer:
    """
    Class to render a quantum circuit using matplotlib.

    Parameters
    ----------
    qc : QuantumCircuit Object
        The quantum circuit to be rendered.

    ax : Axes Object, optional
        The axes object to plot the circuit. The default is None.

    **style
        Additional style arguments to be passed to the `StyleConfig` dataclass.
    """

    def __init__(
        self,
        qc: QubitCircuit,
        ax: Axes = None,
        **style,
    ) -> None:

        self._qc = qc
        self._ax = ax
        self._qwires = qc.N
        self._cwires = qc.num_cbits

        self._cwire_sep = 0.02
        self._min_gate_height = 0.2
        self._min_gate_width = 0.2
        self._default_layers = 2
        self._arrow_lenght = 0.06
        self._connector_r = 0.01
        self._target_node_r = 0.12
        self._control_node_r = 0.05
        self._display_layer_len = 0
        self._start_pad = 0.1
        self._layer_list = {i: [self._start_pad] for i in range(self._qwires)}

        # user defined style
        style = {} if style is None else style
        self.style = StyleConfig(**style)

        # fig config
        self._zorder = {
            "wire": 1,
            "wire_label": 1,
            "gate": 2,
            "node": 2,
            "bridge": 2,
            "connector": 3,
            "gate_label": 3,
            "node_label": 3,
        }
        self.style.fig_height = (
            (
                (self._qwires + self._cwires)
                * self.style.wire_sep
                * 0.393701
                * 3
            )
            if self.style.fig_height is None
            else self.style.fig_height
        )
        if self._ax is None:
            self.fig, self._ax = plt.subplots(
                figsize=(self.style.fig_width, self.style.fig_height),
                dpi=self.style.dpi,
                facecolor=self.style.bgcolor,
            )

    def _get_xskip(self, wire_list: List[int], layer: int) -> float:
        """
        Get the xskip (horizontal value for getting to requested layer) for the gate to be plotted.

        Parameters
        ----------
        wire_list : list
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.
        """

        if self.style.align_layer:
            wire_list = list(range(self._qwires))

        xskip = []
        for wire in wire_list:
            xskip.append(sum(self._layer_list[wire][:layer]))

        return max(xskip)

    def _get_text_width(
        self,
        text: str,
        fontsize: float,
        fontweight: Union[float, str],
        fontfamily: str,
        fontstyle: str,
    ) -> float:
        """
        Get the width of the text to be plotted.

        Parameters
        ----------
        text : str
            The text to be plotted.

        fontsize : float
            The fontsize of the text.

        fontweight : str or float
            The fontweight of the text.

        fontfamily : str
            The fontfamily of the text.

        fontstyle : str
            The fontstyle of the text.
        Returns
        -------
        float
            The width of the text in cm.
        """

        text_obj = plt.Text(
            0,
            0,
            text,
            fontsize=fontsize,
            fontweight=fontweight,
            fontfamily=fontfamily,
            fontstyle=fontstyle,
        )
        self._ax.add_artist(text_obj)

        bbox = text_obj.get_window_extent(
            renderer=self._ax.figure.canvas.get_renderer()
        )
        inv = self._ax.transData.inverted()
        bbox_data = bbox.transformed(inv)
        text_obj.remove()

        return bbox_data.width * 2.54 * 3

    def _manage_layers(
        self,
        gate_width: float,
        wire_list: List[int],
        layer: int,
        xskip: float = 0,
    ) -> None:
        """
        Manages and updates the layer widths according to the gate's width just plotted.

        Parameters
        ----------
        gate_width : float
            The width of the gate to be plotted.

        wire_list : list
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.

        xskip : float, optional
            The horizontal value for getting to requested layer. The default is 0.
        """

        for wire in wire_list:
            # check if requested layer exists for the wire
            if len(self._layer_list[wire]) > layer:
                # check if the layer width is greater than new layer width
                if (
                    self._layer_list[wire][layer]
                    < gate_width + self.style.gate_margin * 2
                ):
                    # update with new layer width
                    self._layer_list[wire][layer] = (
                        gate_width + self.style.gate_margin * 2
                    )
            else:
                # add layer width: new layer width + missing layer widths if exits
                temp = xskip - sum(self._layer_list[wire]) if xskip != 0 else 0
                self._layer_list[wire].append(
                    temp + gate_width + self.style.gate_margin * 2
                )

    def _add_wire(self) -> None:
        """
        Adds the wires to the circuit.
        """
        max_len = (
            max([sum(self._layer_list[i]) for i in range(self._qwires)])
            + self.style.end_wire_ext * self.style.layer_sep
        )

        for i in range(self._qwires):
            wire = plt.Line2D(
                [0, max_len],
                [
                    (i + self._cwires) * self.style.wire_sep,
                    (i + self._cwires) * self.style.wire_sep,
                ],
                lw=1,
                color=self.style.wire_color,
                zorder=self._zorder["wire"],
            )
            self._ax.add_line(wire)

        for i in range(self._cwires):
            wire_up = plt.Line2D(
                [0, max_len],
                [
                    (i * self.style.wire_sep) + self._cwire_sep,
                    (i * self.style.wire_sep) + self._cwire_sep,
                ],
                lw=1,
                color=self.style.wire_color,
                zorder=self._zorder["wire"],
            )
            wire_down = plt.Line2D(
                [0, max_len],
                [
                    (i * self.style.wire_sep) - self._cwire_sep,
                    (i * self.style.wire_sep) - self._cwire_sep,
                ],
                lw=1,
                color=self.style.wire_color,
                zorder=self._zorder["wire"],
            )
            self._ax.add_line(wire_up)
            self._ax.add_line(wire_down)

    def _add_wire_labels(self) -> None:
        """
        Adds the wire labels to the circuit.
        """

        if self.style.wire_label is None:
            default_labels = [f"$c_{{{i}}}$" for i in range(self._cwires)] + [
                f"$q_{{{i}}}$" for i in range(self._qwires)
            ]
            self.style.wire_label = default_labels

        self.max_label_width = max(
            [
                self._get_text_width(
                    label,
                    self.style.fontsize,
                    "normal",
                    "monospace",
                    "normal",
                )
                for label in self.style.wire_label
            ]
        )

        for i, label in enumerate(self.style.wire_label):
            wire_label = plt.Text(
                -self.style.label_pad,
                i * self.style.wire_sep,
                label,
                fontsize=self.style.fontsize,
                fontweight="normal",
                fontfamily="monospace",
                fontstyle="normal",
                verticalalignment="center",
                horizontalalignment="right",
                zorder=self._zorder["wire_label"],
                color=self.style.wire_color,
            )
            self._ax.add_artist(wire_label)

    def _draw_control_node(self, pos: int, xskip: float, color: str) -> None:
        """
        Draw the control node for the multi-qubit gate.

        Parameters
        ----------
        pos : int
            The position of the control node, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node. HEX code or color name supported by matplotlib are valid.
        """

        pos = pos + self._cwires

        control_node = Circle(
            (
                xskip + self.style.gate_margin + self.style.gate_pad,
                pos * self.style.wire_sep,
            ),
            self._control_node_r,
            color=color,
            zorder=self._zorder["node"],
        )
        self._ax.add_artist(control_node)

    def _draw_target_node(self, pos: int, xskip: float, color: str) -> None:
        """
        Draw the target node for the multi-qubit gate.

        Parameters
        ----------
        pos : int
            The position of the target node, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node. HEX code or color name supported by matplotlib are valid.
        """

        pos = pos + self._cwires

        target_node = Circle(
            (
                xskip + self.style.gate_margin + self.style.gate_pad,
                pos * self.style.wire_sep,
            ),
            self._target_node_r,
            color=color,
            zorder=self._zorder["node"],
        )
        vertical_line = plt.Line2D(
            (
                xskip + self.style.gate_margin + self.style.gate_pad,
                xskip + self.style.gate_margin + self.style.gate_pad,
            ),
            (
                pos * self.style.wire_sep - self._target_node_r / 2,
                pos * self.style.wire_sep + self._target_node_r / 2,
            ),
            lw=1.5,
            color=self.style.color,
            zorder=self._zorder["node_label"],
        )
        horizontal_line = plt.Line2D(
            (
                xskip
                + self.style.gate_margin
                + self.style.gate_pad
                - self._target_node_r / 2,
                xskip
                + self.style.gate_margin
                + self.style.gate_pad
                + self._target_node_r / 2,
            ),
            (pos * self.style.wire_sep, pos * self.style.wire_sep),
            lw=1.5,
            color=self.style.color,
            zorder=self._zorder["node_label"],
        )

        self._ax.add_artist(target_node)
        self._ax.add_line(vertical_line)
        self._ax.add_line(horizontal_line)

    def _draw_qbridge(
        self, pos1: int, pos2: int, xskip: float, color: str
    ) -> None:
        """
        Draw the bridge between the control and target nodes for the multi-qubit gate.

        Parameters
        ----------
        pos1 : int
            The position of the first node for the bridge, in terms of the wire number.

        pos2 : int
            The position of the second node for the bridge, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node. HEX code or color name supported by matplotlib are valid.
        """
        pos2 = pos2 + self._cwires
        pos1 = pos1 + self._cwires

        bridge = plt.Line2D(
            [
                xskip + self.style.gate_margin + self.style.gate_pad,
                xskip + self.style.gate_margin + self.style.gate_pad,
            ],
            [pos1 * self.style.wire_sep, pos2 * self.style.wire_sep],
            color=color,
            zorder=self._zorder["bridge"],
        )
        self._ax.add_line(bridge)

    def _draw_cbridge(
        self, c_pos: int, q_pos: int, xskip: float, color: str
    ) -> None:
        """
        Draw the bridge between the classical and quantum wires for the measurement gate.

        Parameters
        ----------
        c_pos : int
            The position of the classical wire, in terms of the wire number.

        q_pos : int
            The position of the quantum wire, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the bridge.
        """
        q_pos = q_pos + self._cwires

        cbridge_l = plt.Line2D(
            (
                xskip
                + self.style.gate_margin
                + self._min_gate_width / 2
                - self._cwire_sep,
                xskip
                + self.style.gate_margin
                + self._min_gate_width / 2
                - self._cwire_sep,
            ),
            (
                c_pos * self.style.wire_sep + self._arrow_lenght,
                q_pos * self.style.wire_sep,
            ),
            color=color,
            zorder=self._zorder["bridge"],
        )
        cbridge_r = plt.Line2D(
            (
                xskip
                + self.style.gate_margin
                + self._min_gate_width / 2
                + self._cwire_sep,
                xskip
                + self.style.gate_margin
                + self._min_gate_width / 2
                + self._cwire_sep,
            ),
            (
                c_pos * self.style.wire_sep + self._arrow_lenght,
                q_pos * self.style.wire_sep,
            ),
            color=color,
            zorder=self._zorder["bridge"],
        )
        end_arrow = FancyArrow(
            xskip + self.style.gate_margin + self._min_gate_width / 2,
            c_pos * self.style.wire_sep + self._arrow_lenght,
            0,
            -self._cwire_sep * 3,
            width=0,
            head_width=self._cwire_sep * 5,
            head_length=self._cwire_sep * 3,
            length_includes_head=True,
            color=color,
            zorder=self._zorder["bridge"],
        )
        self._ax.add_line(cbridge_l)
        self._ax.add_line(cbridge_r)
        self._ax.add_artist(end_arrow)

    def _draw_swap_mark(self, pos: int, xskip: int, color: str) -> None:
        """
        Draw the swap mark for the SWAP gate.

        Parameters
        ----------
        pos : int
            The position of the swap mark, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the swap mark.
        """

        pos = pos + self._cwires

        dia_left = plt.Line2D(
            [
                xskip
                + self.style.gate_margin
                + self.style.gate_pad
                + self._min_gate_width / 3,
                xskip
                + self.style.gate_margin
                + self.style.gate_pad
                - self._min_gate_width / 3,
            ],
            [
                pos * self.style.wire_sep + self._min_gate_height / 2,
                pos * self.style.wire_sep - self._min_gate_height / 2,
            ],
            color=color,
            linewidth=2,
            zorder=self._zorder["gate"],
        )
        dia_right = plt.Line2D(
            [
                xskip
                + self.style.gate_margin
                + self.style.gate_pad
                - self._min_gate_width / 3,
                xskip
                + self.style.gate_margin
                + self.style.gate_pad
                + self._min_gate_width / 3,
            ],
            [
                pos * self.style.wire_sep + self._min_gate_height / 2,
                pos * self.style.wire_sep - self._min_gate_height / 2,
            ],
            color=color,
            linewidth=2,
            zorder=self._zorder["gate"],
        )
        self._ax.add_line(dia_left)
        self._ax.add_line(dia_right)

    def to_pi_fraction(self, value: float, tolerance: float = 0.01) -> str:
        """
        Convert a value to a string fraction of pi.

        Parameters
        ----------
        value : float
            The value to be converted.

        tolerance : float, optional
            The tolerance for the fraction. The default is 0.01.

        Returns
        -------
        str
            The value in terms of pi.
        """

        pi_value = value / np.pi
        if abs(pi_value - round(pi_value)) < tolerance:
            num = round(pi_value)
            return f"[{num}\\pi]" if num != 1 else "[\\pi]"

        for denom in [2, 3, 4, 6, 8, 12]:
            fraction_value = pi_value * denom
            if abs(fraction_value - round(fraction_value)) < tolerance:
                num = round(fraction_value)
                return (
                    f"[{num}\\pi/{denom}]" if num != 1 else f"[\\pi/{denom}]"
                )

        return f"[{round(value, 2)}]"

    def _draw_singleq_gate(self, gate: Gate, layer: int) -> None:
        """
        Draw the single qubit gate.

        Parameters
        ----------
        gate : Gate Object
            The gate to be plotted.

        layer : int
            The layer the gate is acting on.
        """

        gate_wire = gate.targets[0]
        if gate.arg_value is not None and self.showarg is True:
            pi_frac = self.to_pi_fraction(gate.arg_value)
            text = f"${{{self.text}}}_{{{pi_frac}}}$"
        else:
            text = self.text

        text_width = self._get_text_width(
            text,
            self.fontsize,
            self.fontweight,
            self.fontfamily,
            self.fontstyle,
        )
        gate_width = max(
            text_width + self.style.gate_pad * 2, self._min_gate_width
        )

        gate_text = plt.Text(
            self._get_xskip([gate_wire], layer)
            + self.style.gate_margin
            + gate_width / 2,
            (gate_wire + self._cwires) * self.style.wire_sep,
            text,
            color=self.fontcolor,
            fontsize=self.fontsize,
            fontweight=self.fontweight,
            fontfamily=self.fontfamily,
            fontstyle=self.fontstyle,
            verticalalignment="center",
            horizontalalignment="center",
            zorder=self._zorder["gate_label"],
        )
        gate_patch = FancyBboxPatch(
            (
                self._get_xskip([gate_wire], layer) + self.style.gate_margin,
                (gate_wire + self._cwires) * self.style.wire_sep
                - self._min_gate_height / 2,
            ),
            gate_width,
            self._min_gate_height,
            boxstyle=self.style.bulge,
            mutation_scale=0.3,
            facecolor=self.color,
            edgecolor=self.color,
            zorder=self._zorder["gate"],
        )

        self._ax.add_artist(gate_text)
        self._ax.add_patch(gate_patch)
        self._manage_layers(gate_width, [gate_wire], layer)

    def _draw_multiq_gate(self, gate: Gate, layer: int) -> None:
        """
        Draw the multi-qubit gate.

        Parameters
        ----------
        gate : Gate Object
            The gate to be plotted.

        layer : int
            The layer the gate is acting on.
        """

        wire_list = list(
            range(self.merged_wires[0], self.merged_wires[-1] + 1)
        )
        com_xskip = self._get_xskip(wire_list, layer)

        if gate.name == "CNOT" or gate.name == "CX":
            self._draw_control_node(gate.controls[0], com_xskip, self.color)
            self._draw_target_node(gate.targets[0], com_xskip, self.color)
            self._draw_qbridge(
                gate.targets[0], gate.controls[0], com_xskip, self.color
            )
            self._manage_layers(
                2 * self.style.gate_pad + self._target_node_r / 3,
                wire_list,
                layer,
                com_xskip,
            )

        elif gate.name == "SWAP":
            self._draw_swap_mark(gate.targets[0], com_xskip, self.color)
            self._draw_swap_mark(gate.targets[1], com_xskip, self.color)
            self._draw_qbridge(
                gate.targets[0], gate.targets[1], com_xskip, self.color
            )
            self._manage_layers(
                2 * (self.style.gate_pad + self._min_gate_width / 3),
                wire_list,
                layer,
                com_xskip,
            )

        elif gate.name == "TOFFOLI":
            self._draw_control_node(gate.controls[0], com_xskip, self.color)
            self._draw_control_node(gate.controls[1], com_xskip, self.color)
            self._draw_target_node(gate.targets[0], com_xskip, self.color)
            self._draw_qbridge(
                gate.targets[0], gate.controls[0], com_xskip, self.color
            )
            self._draw_qbridge(
                gate.targets[0], gate.controls[1], com_xskip, self.color
            )
            self._manage_layers(
                2 * self.style.gate_pad + self._target_node_r / 3,
                wire_list,
                layer,
                com_xskip,
            )

        else:

            adj_targets = [i + self._cwires for i in sorted(gate.targets)]
            text_width = self._get_text_width(
                self.text,
                self.fontsize,
                self.fontweight,
                self.fontfamily,
                self.fontstyle,
            )
            gate_width = max(
                text_width + self.style.gate_pad * 2, self._min_gate_width
            )
            xskip = self._get_xskip(wire_list, layer)

            gate_text = plt.Text(
                xskip + self.style.gate_margin + gate_width / 2,
                (adj_targets[0] + adj_targets[-1]) / 2 * self.style.wire_sep,
                self.text,
                color=self.fontcolor,
                fontsize=self.fontsize,
                fontweight=self.fontweight,
                fontfamily=self.fontfamily,
                fontstyle=self.fontstyle,
                verticalalignment="center",
                horizontalalignment="center",
                zorder=self._zorder["gate_label"],
            )

            gate_patch = FancyBboxPatch(
                (
                    xskip + self.style.gate_margin,
                    adj_targets[0] * self.style.wire_sep
                    - self._min_gate_height / 2,
                ),
                gate_width,
                self._min_gate_height
                + self.style.wire_sep * (adj_targets[-1] - adj_targets[0]),
                boxstyle=self.style.bulge,
                mutation_scale=0.3,
                facecolor=self.color,
                edgecolor=self.color,
                zorder=self._zorder["gate"],
            )

            if len(gate.targets) > 1:
                for i in range(len(gate.targets)):
                    connector_l = Circle(
                        (
                            xskip + self.style.gate_margin - self._connector_r,
                            (adj_targets[i]) * self.style.wire_sep,
                        ),
                        self._connector_r,
                        color=self.fontcolor,
                        zorder=self._zorder["connector"],
                    )
                    connector_r = Circle(
                        (
                            xskip
                            + self.style.gate_margin
                            + gate_width
                            + self._connector_r,
                            (adj_targets[i]) * self.style.wire_sep,
                        ),
                        self._connector_r,
                        color=self.fontcolor,
                        zorder=self._zorder["connector"],
                    )
                    self._ax.add_artist(connector_l)
                    self._ax.add_artist(connector_r)

            # add cbridge if control qubits are present
            if gate.controls is not None:
                for control in gate.controls:
                    self._draw_control_node(
                        control, xskip + text_width / 2, self.color
                    )
                    self._draw_qbridge(
                        control,
                        gate.targets[0],
                        xskip + text_width / 2,
                        self.color,
                    )

            self._ax.add_artist(gate_text)
            self._ax.add_patch(gate_patch)
            self._manage_layers(gate_width, wire_list, layer, xskip)

    def _draw_measure(self, c_pos: int, q_pos: int, layer: int) -> None:
        """
        Draw the measurement gate.

        Parameters
        ----------
        c_pos : int
            The position of the classical wire.

        q_pos : int
            The position of the quantum wire.

        layer : int
            The layer the gate is acting on.
        """

        xskip = self._get_xskip(
            list(range(0, self.merged_wires[-1] + 1)), layer
        )
        measure_box = FancyBboxPatch(
            (
                xskip + self.style.gate_margin,
                (q_pos + self._cwires) * self.style.wire_sep
                - self._min_gate_height / 2,
            ),
            self._min_gate_width,
            self._min_gate_height,
            boxstyle=self.style.bulge,
            mutation_scale=0.3,
            facecolor=self.style.bgcolor,
            edgecolor=self.style.measure_color,
            linewidth=1.25,
            zorder=self._zorder["gate"],
        )
        arc = Arc(
            (
                xskip + self.style.gate_margin + self._min_gate_width / 2,
                (q_pos + self._cwires) * self.style.wire_sep
                - self._min_gate_height / 2,
            ),
            self._min_gate_width * 1.5,
            self._min_gate_height * 1,
            angle=0,
            theta1=0,
            theta2=180,
            color=self.style.measure_color,
            linewidth=1.25,
            zorder=self._zorder["gate_label"],
        )
        arrow = FancyArrow(
            xskip + self.style.gate_margin + self._min_gate_width / 2,
            (q_pos + self._cwires) * self.style.wire_sep
            - self._min_gate_height / 2,
            self._min_gate_width * 0.7,
            self._min_gate_height * 0.7,
            length_includes_head=True,
            head_width=0,
            linewidth=1.25,
            color=self.style.measure_color,
            zorder=self._zorder["gate_label"],
        )

        self._draw_cbridge(c_pos, q_pos, xskip, color=self.style.measure_color)
        self._manage_layers(
            self._min_gate_width,
            list(range(0, self.merged_wires[-1] + 1)),
            layer,
            xskip,
        )
        self._ax.add_patch(measure_box)
        self._ax.add_artist(arc)
        self._ax.add_artist(arrow)

    def canvas_plot(self) -> None:
        """
        Plot the quantum circuit.
        """

        self._add_wire_labels()

        for gate in self._qc.gates:

            if isinstance(gate, Measurement):
                self.merged_wires = gate.targets.copy()
                self.merged_wires.sort()

                self._draw_measure(
                    gate.classical_store,
                    gate.targets[0],
                    max(
                        len(self._layer_list[i])
                        for i in range(0, self.merged_wires[-1] + 1)
                    ),
                )

            if isinstance(gate, Gate):
                style = gate.style if gate.style is not None else {}
                self.text = (
                    gate.arg_label if gate.arg_label is not None else gate.name
                )
                self.color = style.get(
                    "color",
                    self.style.theme.get(
                        gate.name, self.style.theme["default_gate"]
                    ),
                )
                self.fontsize = style.get("fontsize", self.style.fontsize)
                self.fontcolor = style.get("fontcolor", self.style.color)
                self.fontweight = style.get("fontweight", "normal")
                self.fontstyle = style.get("fontstyle", "normal")
                self.fontfamily = style.get("fontfamily", "monospace")
                self.showarg = style.get("showarg", False)

                # multi-qubit gate
                if (
                    len(gate.targets) > 1
                    or getattr(gate, "controls", False) is not None
                ):
                    self.merged_wires = gate.targets.copy()
                    if gate.controls is not None:
                        self.merged_wires += gate.controls.copy()
                    self.merged_wires.sort()

                    find_layer = [
                        len(self._layer_list[i])
                        for i in range(
                            self.merged_wires[0], self.merged_wires[-1] + 1
                        )
                    ]
                    self._draw_multiq_gate(gate, max(find_layer))

                else:
                    self._draw_singleq_gate(
                        gate, len(self._layer_list[gate.targets[0]])
                    )

        self._add_wire()
        self._fig_config()
        plt.show()

    def _fig_config(self) -> None:
        """
        Configure the figure settings.
        """

        self._ax.set_ylim(
            -self.style.padding,
            self.style.padding
            + (self._qwires + self._cwires - 1) * self.style.wire_sep,
        )
        self._ax.set_xlim(
            -self.style.padding - self.max_label_width - self.style.label_pad,
            self.style.padding
            + self.style.end_wire_ext * self.style.layer_sep
            + max([sum(self._layer_list[i]) for i in range(self._qwires)]),
        )
        if self.style.title is not None:
            self._ax.set_title(
                self.style.title, pad=10, color=self.style.wire_color
            )
        self._ax.set_aspect("equal")
        self._ax.axis("off")

    def save(self, filename: str, **kwargs) -> None:
        """
        Save the circuit to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the circuit to.

        **kwargs
            Additional arguments to be passed to `plt.savefig`.
        """

        self.fig.savefig(filename, **kwargs)
