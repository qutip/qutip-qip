"""
Module for rendering a quantum circuit in text format.
"""

import math

from ..operations import Measurement
from ..circuit import QubitCircuit

__all__ = [
    "TextRenderer",
]


class TextRenderer:
    """
    A class to render a quantum circuit in text format.
    """

    def __init__(self, qc: QubitCircuit):

        self._qc = qc
        self._qwires = qc.N
        self._cwires = qc.num_cbits
        self.gate_pad = 1

        self._render_strs = {
            "top_lid": ["  "] * self._qwires + ["  "] * self._cwires,
            "mid": ["──"] * self._qwires + ["══"] * self._cwires,
            "bot_lid": ["  "] * self._qwires + ["  "] * self._cwires,
        }

        self._layer_list = {i: [] for i in range(self._qwires + self._cwires)}

    def get_xskip(self, wire_list, layer):
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
            xskip.append(sum(self._layer_list[wire][:layer]))

        return max(xskip)

    def manage_layers(self, wire_list, layer, xskip, gate_width):
        """
        Manages and updates the layer widths according to the gate'lid_seg width just plotted.

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
            if len(self._layer_list[wire]) > layer:
                if self._layer_list[wire][layer] < gate_width:
                    self._layer_list[wire][layer] = gate_width
            else:
                temp = xskip - sum(self._layer_list[wire]) if xskip != 0 else 0
                self._layer_list[wire].append(temp + gate_width)

    def add_wire_labels(self):
        """
        Add wire labels to the circuit
        """

        default_labels = [f"q{i}" for i in range(self._qwires)] + [
            f"c{i}" for i in range(self._cwires)
        ]
        for i, label in enumerate(default_labels):
            self._render_strs["mid"][i] = (
                f" {label} :" + self._render_strs["mid"][i]
            )
            self._render_strs["top_lid"][i] = " " * (len(f" {label} :") + 2)
            self._render_strs["bot_lid"][i] = " " * (len(f" {label} :") + 2)

        self._layer_list = {
            i: [len(self._render_strs["mid"][i])]
            for i in range(self._qwires + self._cwires)
        }

    def _extend_line(self):
        """
        Extend all the wires to the same length
        """

        length_wire = [
            sum(self._layer_list[i])
            for i in range(self._qwires + self._cwires)
        ]
        max_length = max(length_wire)

        for i in range(self._qwires):
            if length_wire[i] < max_length:
                diff = max_length - length_wire[i]
                self._render_strs["top_lid"][i] += " " * diff
                self._render_strs["mid"][i] += "─" * diff
                self._render_strs["bot_lid"][i] += " " * diff

        for i in range(self._cwires):
            if length_wire[i + self._qwires] < max_length:
                diff = max_length - length_wire[i + self._qwires]
                self._render_strs["top_lid"][i + self._qwires] += " " * diff
                self._render_strs["mid"][i + self._qwires] += "═" * diff
                self._render_strs["bot_lid"][i + self._qwires] += " " * diff

    def _draw_singleq_gate(self, gate):
        """
        Draw a single qubit gate

        Parameters
        ----------
        gate : Gate
            The gate to be drawn.

        Returns
        -------
        tuple
            The parts of the gate to be drawn. The parts are the top_lid, mid, and bot_lid.

        int
            The width of the gate.
        """

        lid_seg = "─" * (self.gate_pad * 2 + len(gate.name))
        pad = " " * self.gate_pad

        top_lid = f" ┌{lid_seg}┐ "
        mid = f"─┤{pad}{gate.name}{pad}├─"
        bot_lid = f" └{lid_seg}┘ "

        return (top_lid, mid, bot_lid), len(top_lid)

    def _draw_multiq_gate(self, gate):
        """
        Draw a multi qubit gate

        Parameters
        ----------
        gate : Gate
            The gate to be drawn.

        Returns
        -------
        tuple
            The parts of the gate to be drawn. The parts are the top_lid, mid, connect, connect_label, and bot_lid.

        int
            The width of the gate.
        """

        lid_seg = "─" * (self.gate_pad * 2 + len(gate.name))
        pad = " " * self.gate_pad

        top_lid = f" ┌{lid_seg}┐ "
        bot_lid = f" └{lid_seg}┘ "
        mid = f" |{pad}{' ' * len(gate.name)}{pad}| "
        connect = f"─┤{pad}{' ' * len(gate.name)}{pad}├─"
        connect_label = f"─┤{pad}{gate.name}{pad}├─"

        # Adjust top_lid or bottom if there is a control wire
        if gate.controls:
            mid_index = len(bot_lid) // 2
            if gate.controls[0] < gate.targets[0]:
                bot_lid = bot_lid[:mid_index] + "┬" + bot_lid[mid_index + 1 :]
            else:
                top_lid = top_lid[:mid_index] + "┴" + top_lid[mid_index + 1 :]

        return (top_lid, mid, connect, connect_label, bot_lid), len(top_lid)

    def _adjust_layer(self, xskip, wire_list):
        """
        Adjust the layers by filling the empty spaces with respective characters
        """

        for wire in wire_list:
            self._render_strs["top_lid"][wire] += " " * (
                xskip - len(self._render_strs["top_lid"][wire])
            )
            self._render_strs["mid"][wire] += "─" * (
                xskip - len(self._render_strs["mid"][wire])
            )
            self._render_strs["bot_lid"][wire] += " " * (
                xskip - len(self._render_strs["bot_lid"][wire])
            )

    def _update_singleq(self, wire_list, parts):
        """
        Update the render strings for single qubit gates
        """
        top_lid, mid, bot_lid = parts
        for wire in wire_list:
            self._render_strs["top_lid"][wire] += top_lid
            self._render_strs["mid"][wire] += mid
            self._render_strs["bot_lid"][wire] += bot_lid

    def _update_target_multiq(self, gate, wire_list, xskip, parts):
        """
        Update the render strings for part of the multi qubit gate drawn on the target wires.

        Parameters
        ----------
        gate : Gate
            The gate to be drawn.

        wire_list : list
            The list of target wires the gate is acting on and all the wires in between.

        xskip : float
            The horizontal value for getting to requested layer.

        parts : tuple
            The parts of the gate to be drawn.
        """

        top_lid, mid, connect, connect_label, bot_lid = parts
        self._adjust_layer(xskip, wire_list)

        for i, wire in enumerate(wire_list):
            if len(gate.targets) == 1:
                self._render_strs["top_lid"][wire] += top_lid
                self._render_strs["mid"][wire] += connect_label
                self._render_strs["bot_lid"][wire] += bot_lid
            elif i == 0 and wire in gate.targets:
                self._render_strs["top_lid"][wire] += mid
                self._render_strs["mid"][wire] += connect_label
                self._render_strs["bot_lid"][wire] += bot_lid
            elif i == len(wire_list) - 1 and wire in gate.targets:
                self._render_strs["top_lid"][wire] += top_lid
                self._render_strs["mid"][wire] += connect
                self._render_strs["bot_lid"][wire] += mid
            else:
                self._render_strs["top_lid"][wire] += mid
                self._render_strs["mid"][wire] += mid
                self._render_strs["bot_lid"][wire] += mid

    def _update_control_multiq(
        self, gate, wire_list_control, xskip, width, is_top_closer
    ):
        """
        Update the render strings for part of the multi qubit gate drawn on the control wires.

        Parameters
        ----------
        gate : Gate
            The gate to be drawn.

        wire_list_control : list
            The list of control wires the gate is acting on and all the wires in between.

        xskip : float
            The horizontal value for getting to requested layer.

        width : int
            The width of the gate.

        is_top_closer : bool
            If the top of the gate is closer to the first control wires.
        """

        bar_conn = " " * (width // 2) + "|" + " " * (width // 2 - 1)
        self._adjust_layer(xskip, wire_list_control)

        for wire in wire_list_control:
            if wire not in gate.targets:
                if wire in gate.controls:
                    self._render_strs["mid"][wire] += (
                        "─" * (math.floor(width / 2))
                        + "▇"
                        + "─" * (math.floor(width / 2) - 1)
                    )
                    self._render_strs["top_lid"][wire] += (
                        bar_conn if is_top_closer else " " * len(bar_conn)
                    )
                    self._render_strs["bot_lid"][wire] += (
                        bar_conn if not is_top_closer else " " * len(bar_conn)
                    )
                else:
                    self._render_strs["top_lid"][wire] += bar_conn
                    self._render_strs["mid"][wire] += bar_conn
                    self._render_strs["bot_lid"][wire] += bar_conn

    def layout(self):
        """
        Layout the circuit
        """
        self.add_wire_labels()

        for gate in self._qc.gates:
            if len(gate.targets) == 1 and gate.controls is None:
                wire_list = gate.targets
                layer = max(len(self._layer_list[i]) for i in wire_list)
                xskip = self.get_xskip(wire_list, layer)

                parts, width = self._draw_singleq_gate(gate)
                self.manage_layers(wire_list, layer, xskip, width)
                self._update_singleq(wire_list, parts)
            else:
                merged_wire = sorted(gate.targets + (gate.controls or []))
                wire_list = list(range(merged_wire[0], merged_wire[-1] + 1))
                layer = max(len(self._layer_list[i]) for i in wire_list)
                xskip = self.get_xskip(wire_list, layer)
                parts, width = self._draw_multiq_gate(gate)
                self.manage_layers(wire_list, layer, xskip, width)

                sorted_targets = sorted(gate.targets)
                wire_list_target = list(
                    range(sorted_targets[0], sorted_targets[-1] + 1)
                )
                self._update_target_multiq(
                    gate, wire_list_target, xskip, parts
                )

                if gate.controls:
                    sorted_controls = sorted(gate.controls)
                    is_top_closer = wire_list_target[0] >= sorted_controls[0]
                    closest_pos = (
                        wire_list_target[0]
                        if not is_top_closer
                        else wire_list_target[-1]
                    )

                    wire_list_control = list(
                        range(
                            min(sorted_controls[0], closest_pos),
                            max(sorted_controls[-1], closest_pos) + 1,
                        )
                    )
                    self._update_control_multiq(
                        gate, wire_list_control, xskip, width, is_top_closer
                    )

        self._extend_line()
        self.print_circuit()

    def print_circuit(self):
        """
        Print the circuit
        """

        for i in range(self._qwires + self._cwires - 1, -1, -1):
            print(self._render_strs["top_lid"][i])
            print(self._render_strs["mid"][i])
            print(self._render_strs["bot_lid"][i])
