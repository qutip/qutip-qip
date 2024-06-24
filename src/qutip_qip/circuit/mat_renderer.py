"""
Module for rendering a quantum circuit using matplotlib library.
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch,
    Circle,
    Arc,
    FancyArrow,
)

from ..operations import Gate, Measurement
from ..circuit import QubitCircuit
from .color_theme import default_theme

__all__ = [
    "MatRenderer",
]


class MatRenderer:
    """
    Class to render a quantum circuit using matplotlib.

    Parameters
    ----------
    qc : QuantumCircuit Object
        The quantum circuit to be rendered.

    padding : float, optional
        The padding around the circuit. The default is 0.3.

    dpi : int, optional
        The resolution of the image. The default is 150.

    bgcolor : str, optional
        The background color of the plotted circuit. The default is "#FFFFFF" (white).

    wire_labels : list, optional
        The labels for the wires.
    """

    def __init__(
        self,
        qc: QubitCircuit,
        padding=0.3,
        dpi: int = 150,
        bgcolor: str = "#FFFFFF",
        wire_label: List[str] = None,
        condense=0.15,
        bulge=True,
    ) -> None:

        self.wire_sep = 0.5
        self.layer_sep = 0.5
        self.cwire_sep = 0.02
        self.gate_height = 0.2
        self.gate_width = 0.2
        self.gate_pad = 0.05
        self.label_pad = 0.2
        self.font_size = 10
        self.default_layers = 2
        self.arrow_lenght = 0.06
        self.connector_r = 0.01
        self.target_node_r = 0.12
        self.display_layer_len = 0
        self.start_pad = 0.1

        self.end_wire_ext = 2
        if bulge:
            self.bulge = "round4"
        else:
            self.bulge = "square"

        self.qc = qc
        self.qwires = qc.N
        self.cwires = qc.num_cbits
        self.dpi = dpi
        self.gate_margin = condense
        self.padding = padding
        self.bgcolor = bgcolor
        self.wire_label = wire_label
        self.layer_list = {i: [self.start_pad] for i in range(self.qwires)}

        # fig config
        self.fig_height = (
            (self.qwires + self.cwires) * self.wire_sep * 0.393701 * 3
        )
        self.fig_width = 10
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            dpi=self.dpi,
            facecolor=self.bgcolor,
        )

        self.canvas_plot()

    def _get_xskip(self, wire_list, layer):
        """
        Get the xskip (horizontal value for getting to requested layer) for the gate to be plotted.

        Parameters
        ----------
        wire_list : list
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.
        """

        xskip = []
        for wire in wire_list:
            xskip.append(sum(self.layer_list[wire][:layer]))

        return max(xskip)

    def _get_text_width(
        self, text, fontsize, fontweight, fontfamily, fontstyle
    ):
        """
        Get the width of the text to be plotted.

        Parameters
        ----------
        text : str
            The text to be plotted.

        fontsize : int
            The fontsize of the text.

        fontweight : str
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
        self.ax.add_artist(text_obj)

        bbox = text_obj.get_window_extent(
            renderer=self.ax.figure.canvas.get_renderer()
        )
        inv = self.ax.transData.inverted()
        bbox_data = bbox.transformed(inv)
        text_obj.remove()

        return bbox_data.width * 2.54 * 3

    def _manage_layers(self, gate_width, wire_list, layer, xskip=0):
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
            if len(self.layer_list[wire]) > layer:
                if (
                    self.layer_list[wire][layer]
                    < gate_width + self.gate_margin * 2
                ):
                    self.layer_list[wire][layer] = (
                        gate_width + self.gate_margin * 2
                    )
            else:
                temp = xskip - sum(self.layer_list[wire]) if xskip != 0 else 0
                self.layer_list[wire].append(
                    temp + gate_width + self.gate_margin * 2
                )

    def _extend_wires(self, ext, end=False):
        """
        Extends the wires in the circuit.

        Parameters
        ----------
        ext : int
            The number of layers to extend the wires.

        end : bool, optional
            If True, the wires are extended further distance for end of the circuit.
            The default is False.
        """

        max_layer = max([sum(self.layer_list[i]) for i in range(self.qwires)])

        if (
            self.display_layer_len < max_layer
            or self.display_layer_len == 0
            or end is True
        ):
            if self.cwires != 0:

                ext_cwires_pos = [
                    [
                        [
                            (
                                self.display_layer_len,
                                max_layer + ext * self.layer_sep,
                            ),
                            (
                                (i * self.wire_sep) + self.cwire_sep,
                                (i * self.wire_sep) + self.cwire_sep,
                            ),
                        ],
                        [
                            (
                                self.display_layer_len,
                                max_layer + ext * self.layer_sep,
                            ),
                            (
                                i * self.wire_sep - self.cwire_sep,
                                i * self.wire_sep - self.cwire_sep,
                            ),
                        ],
                    ]
                    for i in range(self.cwires)
                ]

                for pos in ext_cwires_pos:
                    wire_up = plt.Line2D(
                        pos[0][0], pos[0][1], lw=1, color="k", zorder=1
                    )
                    wire_down = plt.Line2D(
                        pos[1][0], pos[1][1], lw=1, color="k", zorder=1
                    )
                    self.ax.add_line(wire_up)
                    self.ax.add_line(wire_down)

            ext_qwires_pos = [
                [
                    (
                        self.display_layer_len,
                        max_layer + ext * self.layer_sep,
                    ),
                    (i * self.wire_sep, i * self.wire_sep),
                ]
                for i in range(self.cwires, self.cwires + self.qwires)
            ]

            for pos in ext_qwires_pos:
                wire = plt.Line2D(pos[0], pos[1], lw=1, color="k", zorder=1)
                self.ax.add_line(wire)

            self.display_layer_len = max_layer

    def _add_wire_labels(self):
        """
        Adds the wire labels to the circuit.
        """

        if self.wire_label is None:
            default_labels = [f"$c_{{{i}}}$" for i in range(self.cwires)] + [
                f"$q_{{{i}}}$" for i in range(self.qwires)
            ]
            self.wire_label = default_labels

        self.max_label_width = max(
            [
                self._get_text_width(
                    label,
                    8,
                    "normal",
                    "monospace",
                    "normal",
                )
                for label in self.wire_label
            ]
        )

        for i, label in enumerate(self.wire_label):
            wire_label = plt.Text(
                -self.label_pad,
                i * self.wire_sep,
                label,
                fontsize=self.font_size,
                verticalalignment="center",
                horizontalalignment="right",
                zorder=3,
            )
            self.ax.add_artist(wire_label)

    def _draw_control_node(self, pos, xskip, color):
        """
        Draw the control node for the multi-qubit gate.

        Parameters
        ----------
        pos : int
            The position of the control node.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node.
        """

        pos = pos + self.cwires

        control_node_radius = 0.05
        control_node = Circle(
            (xskip + self.gate_margin + self.gate_pad, pos * self.wire_sep),
            control_node_radius,
            color=color,
            zorder=2,
        )
        self.ax.add_artist(control_node)

    def _draw_target_node(self, pos, xskip, node_color):
        """
        Draw the target node for the multi-qubit gate.

        Parameters
        ----------
        pos : int
            The position of the target node.

        xskip : float
            The horizontal value for getting to requested layer.

        node_color : str
            The color of the target node.
        """

        pos = pos + self.cwires

        target_node = Circle(
            (xskip + self.gate_margin + self.gate_pad, pos * self.wire_sep),
            self.target_node_r,
            color=node_color,
            zorder=2,
        )
        vertical_line = plt.Line2D(
            (
                xskip + self.gate_margin + self.gate_pad,
                xskip + self.gate_margin + self.gate_pad,
            ),
            (
                pos * self.wire_sep - self.target_node_r / 2,
                pos * self.wire_sep + self.target_node_r / 2,
            ),
            lw=1.5,
            color="white",
            zorder=3,
        )
        horizontal_line = plt.Line2D(
            (
                xskip
                + self.gate_margin
                + self.gate_pad
                - self.target_node_r / 2,
                xskip
                + self.gate_margin
                + self.gate_pad
                + self.target_node_r / 2,
            ),
            (pos * self.wire_sep, pos * self.wire_sep),
            lw=1.5,
            color="white",
            zorder=3,
        )

        self.ax.add_artist(target_node)
        self.ax.add_line(vertical_line)
        self.ax.add_line(horizontal_line)

    def _draw_qbridge(self, pos1, pos2, xskip, color):
        """
        Draw the bridge between the control and target nodes for the multi-qubit gate.

        Parameters
        ----------
        pos1 : int
            The position of the first node for the bridge.

        pos2 : int
            The position of the second node for the bridge.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the bridge.
        """
        pos2 = pos2 + self.cwires
        pos1 = pos1 + self.cwires

        bridge = plt.Line2D(
            [
                xskip + self.gate_margin + self.gate_pad,
                xskip + self.gate_margin + self.gate_pad,
            ],
            [pos1 * self.wire_sep, pos2 * self.wire_sep],
            color=color,
            zorder=2,
        )
        self.ax.add_line(bridge)

    def _draw_cbridge(self, c_pos, q_pos, xskip, color):
        """
        Draw the bridge between the classical and quantum wires for the measurement gate.

        Parameters
        ----------
        c_pos : int
            The position of the classical wire.

        q_pos : int
            The position of the quantum wire.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the bridge.
        """
        q_pos = q_pos + self.cwires

        cbridge_l = plt.Line2D(
            (
                xskip
                + self.gate_margin
                + self.gate_width / 2
                - self.cwire_sep,
                xskip
                + self.gate_margin
                + self.gate_width / 2
                - self.cwire_sep,
            ),
            (c_pos * self.wire_sep + self.arrow_lenght, q_pos * self.wire_sep),
            color=color,
            zorder=2,
        )
        cbridge_r = plt.Line2D(
            (
                xskip
                + self.gate_margin
                + self.gate_width / 2
                + self.cwire_sep,
                xskip
                + self.gate_margin
                + self.gate_width / 2
                + self.cwire_sep,
            ),
            (c_pos * self.wire_sep + self.arrow_lenght, q_pos * self.wire_sep),
            color=color,
            zorder=2,
        )
        end_arrow = FancyArrow(
            xskip + self.gate_margin + self.gate_width / 2,
            c_pos * self.wire_sep + self.arrow_lenght,
            0,
            -self.cwire_sep * 3,
            width=0,
            head_width=self.cwire_sep * 5,
            head_length=self.cwire_sep * 3,
            length_includes_head=True,
            color="k",
            zorder=2,
        )
        self.ax.add_line(cbridge_l)
        self.ax.add_line(cbridge_r)
        self.ax.add_artist(end_arrow)

    def _draw_swap_mark(self, pos, xskip, color):
        """
        Draw the swap mark for the SWAP gate.

        Parameters
        ----------
        pos : int
            The position of the swap mark.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the swap mark.
        """

        pos = pos + self.cwires

        dia_left = plt.Line2D(
            [
                xskip + self.gate_margin + self.gate_pad + self.gate_width / 3,
                xskip + self.gate_margin + self.gate_pad - self.gate_width / 3,
            ],
            [
                pos * self.wire_sep + self.gate_height / 2,
                pos * self.wire_sep - self.gate_height / 2,
            ],
            color=color,
            linewidth=1.5,
            zorder=3,
        )
        dia_right = plt.Line2D(
            [
                xskip + self.gate_margin + self.gate_pad - self.gate_width / 3,
                xskip + self.gate_margin + self.gate_pad + self.gate_width / 3,
            ],
            [
                pos * self.wire_sep + self.gate_height / 2,
                pos * self.wire_sep - self.gate_height / 2,
            ],
            color=color,
            linewidth=1.5,
            zorder=3,
        )
        self.ax.add_line(dia_left)
        self.ax.add_line(dia_right)

    def to_pi_fraction(self, value, tolerance=0.01):
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

    def _draw_singleq_gate(self, gate, layer):
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
        if gate.arg_value is not None:
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
        gate_width = max(text_width + self.gate_pad * 2, self.gate_width)

        gate_text = plt.Text(
            self._get_xskip([gate_wire], layer)
            + self.gate_margin
            + gate_width / 2,
            (gate_wire + self.cwires) * self.wire_sep,
            text,
            color=self.fontcolor,
            fontsize=self.fontsize,
            fontweight=self.fontweight,
            fontfamily=self.fontfamily,
            fontstyle=self.fontstyle,
            verticalalignment="center",
            horizontalalignment="center",
            zorder=3,
        )
        gate_patch = FancyBboxPatch(
            (
                self._get_xskip([gate_wire], layer) + self.gate_margin,
                (gate_wire + self.cwires) * self.wire_sep
                - self.gate_height / 2,
            ),
            gate_width,
            self.gate_height,
            boxstyle=self.bulge,
            mutation_scale=0.3,
            facecolor=self.color,
            edgecolor=self.color,
            zorder=2,
        )

        self.ax.add_artist(gate_text)
        self.ax.add_patch(gate_patch)
        self._manage_layers(gate_width, [gate_wire], layer)

    def _draw_multiq_gate(self, gate, layer):
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
            range(self.merged_qubits[0], self.merged_qubits[-1] + 1)
        )
        com_xskip = self._get_xskip(self.merged_qubits, layer)

        if gate.name == "CNOT":
            self._draw_control_node(gate.controls[0], com_xskip, self.color)
            self._draw_target_node(gate.targets[0], com_xskip, self.color)
            self._draw_qbridge(
                gate.targets[0], gate.controls[0], com_xskip, self.color
            )
            self._manage_layers(
                2 * self.gate_pad + self.target_node_r / 3,
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
                2 * (self.gate_pad + self.gate_width / 3),
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
                2 * self.gate_pad + self.target_node_r / 3,
                wire_list,
                layer,
                com_xskip,
            )

        else:

            adj_targets = [i + self.cwires for i in sorted(gate.targets)]
            text_width = self._get_text_width(
                self.text,
                self.fontsize,
                self.fontweight,
                self.fontfamily,
                self.fontstyle,
            )
            gate_width = max(text_width + self.gate_pad * 2, self.gate_width)
            xskip = self._get_xskip(wire_list, layer)

            gate_text = plt.Text(
                xskip + self.gate_margin + gate_width / 2,
                (adj_targets[0] + adj_targets[-1]) / 2 * self.wire_sep,
                self.text,
                color=self.fontcolor,
                fontsize=self.fontsize,
                fontweight=self.fontweight,
                fontfamily=self.fontfamily,
                fontstyle=self.fontstyle,
                verticalalignment="center",
                horizontalalignment="center",
                zorder=3,
            )

            gate_patch = FancyBboxPatch(
                (
                    xskip + self.gate_margin,
                    adj_targets[0] * self.wire_sep - self.gate_height / 2,
                ),
                gate_width,
                self.gate_height
                + self.wire_sep * (adj_targets[-1] - adj_targets[0]),
                boxstyle=self.bulge,
                mutation_scale=0.3,
                facecolor=self.color,
                edgecolor=self.color,
                zorder=2,
            )

            if len(gate.targets) > 1:
                for i in range(len(gate.targets)):
                    connector_l = Circle(
                        (
                            xskip + self.gate_margin - self.connector_r,
                            (adj_targets[i]) * self.wire_sep,
                        ),
                        self.connector_r,
                        color=self.fontcolor,
                        zorder=3,
                    )
                    connector_r = Circle(
                        (
                            xskip
                            + self.gate_margin
                            + gate_width
                            + self.connector_r,
                            (adj_targets[i]) * self.wire_sep,
                        ),
                        self.connector_r,
                        color=self.fontcolor,
                        zorder=3,
                    )
                    self.ax.add_artist(connector_l)
                    self.ax.add_artist(connector_r)

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

            self.ax.add_artist(gate_text)
            self.ax.add_patch(gate_patch)
            self._manage_layers(gate_width, wire_list, layer, xskip)

            return None

    def _draw_measure(self, c_pos, q_pos, layer):
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
            list(range(0, self.merged_qubits[-1] + 1)), layer
        )
        measure_box = FancyBboxPatch(
            (
                xskip + self.gate_margin,
                (q_pos + self.cwires) * self.wire_sep - self.gate_height / 2,
            ),
            self.gate_width,
            self.gate_height,
            boxstyle=self.bulge,
            mutation_scale=0.3,
            facecolor="white",
            edgecolor="k",
            zorder=2,
        )
        arc = Arc(
            (
                xskip + self.gate_margin + self.gate_width / 2,
                (q_pos + self.cwires) * self.wire_sep - self.gate_height / 2,
            ),
            self.gate_width * 1.5,
            self.gate_height * 1,
            angle=0,
            theta1=0,
            theta2=180,
            color="k",
            zorder=2,
        )
        arrow = FancyArrow(
            xskip + self.gate_margin + self.gate_width / 2,
            (q_pos + self.cwires) * self.wire_sep - self.gate_height / 2,
            self.gate_width * 0.7,
            self.gate_height * 0.7,
            length_includes_head=True,
            head_width=0,
            color="k",
            zorder=2,
        )

        self._draw_cbridge(c_pos, q_pos, xskip, "k")
        self._manage_layers(
            self.gate_width,
            list(range(0, self.merged_qubits[-1] + 1)),
            layer,
            xskip,
        )
        self.ax.add_patch(measure_box)
        self.ax.add_artist(arc)
        self.ax.add_artist(arrow)

    def canvas_plot(self):
        """
        Plot the quantum circuit.
        """

        self._extend_wires(self.default_layers)
        self._add_wire_labels()

        for gate in self.qc.gates:

            if isinstance(gate, Measurement):
                self.merged_qubits = gate.targets.copy()
                self.merged_qubits.sort()

                find_layer = [
                    len(self.layer_list[i])
                    for i in range(0, self.merged_qubits[-1] + 1)
                ]
                self._draw_measure(
                    gate.classical_store,
                    gate.targets[0],
                    max(find_layer),
                )

            if isinstance(gate, Gate):
                style = gate.style if gate.style is not None else {}
                self.text = style.get("text", gate.name)
                self.color = style.get(
                    "color", default_theme.get(gate.name, "k")
                )
                self.fontsize = style.get("fontsize", self.font_size)
                self.fontcolor = style.get("fontcolor", "#FFFFFF")
                self.fontweight = style.get("fontweight", "normal")
                self.fontstyle = style.get("fontstyle", "normal")
                self.fontfamily = style.get("fontfamily", "monospace")

                # multi-qubit gate
                if (
                    len(gate.targets) > 1
                    or getattr(gate, "controls", False) is not None
                ):
                    self.merged_qubits = gate.targets.copy()
                    if gate.controls is not None:
                        self.merged_qubits += gate.controls.copy()
                    self.merged_qubits.sort()

                    find_layer = [
                        len(self.layer_list[i])
                        for i in range(
                            self.merged_qubits[0], self.merged_qubits[-1] + 1
                        )
                    ]
                    self._draw_multiq_gate(gate, max(find_layer))

                else:
                    self._draw_singleq_gate(
                        gate, len(self.layer_list[gate.targets[0]])
                    )

                self._extend_wires(0)

        self._extend_wires(self.end_wire_ext, end=True)
        self._fig_config()

    def _fig_config(self):
        """
        Configure the figure settings.
        """

        self.ax.set_ylim(
            -self.padding,
            self.padding + (self.qwires + self.cwires - 1) * self.wire_sep,
        )
        self.ax.set_xlim(
            -self.padding - self.max_label_width - self.label_pad,
            self.padding
            + 2 * self.layer_sep
            + max([sum(self.layer_list[i]) for i in range(self.qwires)]),
        )
        self.ax.set_aspect("equal")
        self.ax.axis("off")
