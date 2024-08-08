"""
Module for rendering a quantum circuit in text format.
"""

from typing import List

from ..operations import Gate, Measurement
from . import QubitCircuit

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
        self.end_ext = 2

        self._render_strs = {
            "top_frame": ["  "] * (self._qwires + self._cwires),
            "mid_frame": ["──"] * self._qwires + ["══"] * self._cwires,
            "bot_frame": ["  "] * (self._qwires + self._cwires),
        }

        self._layer_list = {i: [] for i in range(self._qwires + self._cwires)}

    def get_xskip(self, wire_list: List[int], layer: int):
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

    def manage_layers(
        self, wire_list: List[int], layer: int, xskip: int, gate_width: int
    ):
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

    def _adjust_layer(self, wire_list: List[int], xskip: int):
        """
        Adjust the layers by filling the empty spaces with respective characters
        """

        for wire in wire_list:
            self._render_strs["top_frame"][wire] += " " * (
                xskip - len(self._render_strs["top_frame"][wire])
            )
            self._render_strs["bot_frame"][wire] += " " * (
                xskip - len(self._render_strs["bot_frame"][wire])
            )
            if wire < self._qwires:
                self._render_strs["mid_frame"][wire] += "─" * (
                    xskip - len(self._render_strs["mid_frame"][wire])
                )
            else:
                self._render_strs["mid_frame"][wire] += "═" * (
                    xskip - len(self._render_strs["mid_frame"][wire])
                )

    def _add_wire_labels(self):
        """
        Add wire labels to the circuit
        """

        default_labels = [f"q{i}" for i in range(self._qwires)] + [
            f"c{i}" for i in range(self._cwires)
        ]
        for i, label in enumerate(default_labels):
            self._render_strs["mid_frame"][i] = (
                f" {label} :" + self._render_strs["mid_frame"][i]
            )
            self._render_strs["top_frame"][i] = " " * (len(f" {label} :") + 2)
            self._render_strs["bot_frame"][i] = " " * (len(f" {label} :") + 2)

        self._layer_list = {
            i: [len(self._render_strs["mid_frame"][i])]
            for i in range(self._qwires + self._cwires)
        }

    def _extend_line(self):
        """
        Extend all the wires to the same length
        """

        wire_len = [sum(wire) for wire in self._layer_list.values()]
        max_len = max(wire_len) + self.end_ext

        for i, length in enumerate(wire_len):
            if length < max_len:
                diff = max_len - length
                if i < self._qwires:
                    self._render_strs["mid_frame"][i] += "─" * diff
                else:
                    self._render_strs["mid_frame"][i] += "═" * diff

    def _draw_singleq_gate(self, gate_name: str):
        """
        Draw a single qubit gate

        Parameters
        ----------
        gate : Gate
            The gate to be drawn.

        Returns
        -------
        tuple
            The parts of the gate to be drawn. The parts are the top_frame, mid_frame, and bot_frame.

        int
            The width of the gate.
        """

        lid_seg = "─" * (self.gate_pad * 2 + len(gate_name))
        pad = " " * self.gate_pad

        top_frame = f" ┌{lid_seg}┐ "
        mid_frame = f"─┤{pad}{gate_name}{pad}├─"
        bot_frame = f" └{lid_seg}┘ "

        return (top_frame, mid_frame, bot_frame), len(top_frame)

    def _draw_multiq_gate(self, gate: Gate):
        """
        Draw a multi qubit gate

        Parameters
        ----------
        gate : Gate
            The gate to be drawn.

        Returns
        -------
        tuple
            The parts of the gate to be drawn.
            i.e. the top_frame, mid_frame, mid_connect, mid_connect_label, and bot_frame.

        int
            The width of the gate.
        """

        lid_seg = "─" * (self.gate_pad * 2 + len(gate.name))
        pad = " " * self.gate_pad

        top_frame = f" ┌{lid_seg}┐ "
        bot_frame = f" └{lid_seg}┘ "
        mid_frame = f" │{pad}{' ' * len(gate.name)}{pad}│ "
        mid_connect = f"─┤{pad}{' ' * len(gate.name)}{pad}├─"
        mid_connect_label = f"─┤{pad}{gate.name}{pad}├─"

        # Adjust top_frame or bottom if there is a control wire
        if gate.controls:
            mid_index = len(bot_frame) // 2
            if gate.controls[0] < gate.targets[0]:
                bot_frame = (
                    bot_frame[:mid_index] + "┬" + bot_frame[mid_index + 1 :]
                )
            else:
                top_frame = (
                    top_frame[:mid_index] + "┴" + top_frame[mid_index + 1 :]
                )

        return (
            top_frame,
            mid_frame,
            mid_connect,
            mid_connect_label,
            bot_frame,
        ), len(top_frame)

    def _draw_measurement_gate(self, measurement: Measurement):
        """
        Draw a measurement gate
        """

        parts, width = self._draw_singleq_gate("M")
        top_frame, mid_frame, bot_frame = parts

        # adjust top_frame or bottom according the placement of the classical wire
        mid_index = len(bot_frame) // 2
        if measurement.classical_store + self._qwires > measurement.targets[0]:
            bot_frame = (
                bot_frame[:mid_index] + "╥" + bot_frame[mid_index + 1 :]
            )
        else:
            top_frame = (
                top_frame[:mid_index] + "╨" + top_frame[mid_index + 1 :]
            )

        return (top_frame, mid_frame, bot_frame), width

    def _update_cbridge(
        self, gate: Gate, wire_list: List[int], xskip: int, width: int
    ):
        """
        Update the render strings for the control bridge

        Parameters
        ----------
        wire_list : list
            The list of control wires the gate is acting on and all the wires in between.

        xskip : float
            The horizontal value for getting to requested layer.

        width : int
            The width of the gate.
        """

        bar_conn = " " * (width // 2) + "║" + " " * (width // 2)
        mid_bar_conn = "─" * (width // 2) + "║" + "─" * (width // 2)
        mid_bar_classical_conn = "═" * (width // 2) + "║" + "═" * (width // 2)
        classical_conn = "═" * (width // 2) + "╩" + "═" * (width // 2)
        self._adjust_layer(wire_list, xskip)

        for wire in wire_list:

            if wire == gate.targets[0]:
                continue

            if wire == self._qwires + gate.classical_store:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["mid_frame"][wire] += classical_conn
                self._render_strs["bot_frame"][wire] += " " * len(bar_conn)
            else:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["bot_frame"][wire] += bar_conn

                # check if the non-store wire is a classical wire or not
                if wire > self._qwires:
                    self._render_strs["mid_frame"][
                        wire
                    ] += mid_bar_classical_conn
                else:
                    self._render_strs["mid_frame"][wire] += mid_bar_conn

    def _update_singleq(self, wire_list, parts):
        """
        Update the render strings for single qubit gates
        """
        top_frame, mid_frame, bot_frame = parts
        for wire in wire_list:
            self._render_strs["top_frame"][wire] += top_frame
            self._render_strs["mid_frame"][wire] += mid_frame
            self._render_strs["bot_frame"][wire] += bot_frame

    def _update_target_multiq(
        self, gate: Gate, wire_list: List[int], xskip: int, parts: List[str]
    ):
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

        top_frame, mid_frame, mid_connect, mid_connect_label, bot_frame = parts
        self._adjust_layer(wire_list, xskip)

        for i, wire in enumerate(wire_list):
            if len(gate.targets) == 1:
                self._render_strs["top_frame"][wire] += top_frame
                self._render_strs["mid_frame"][wire] += mid_connect_label
                self._render_strs["bot_frame"][wire] += bot_frame
            elif i == 0 and wire in gate.targets:
                self._render_strs["top_frame"][wire] += mid_frame
                self._render_strs["mid_frame"][wire] += mid_connect_label
                self._render_strs["bot_frame"][wire] += bot_frame
            elif i == len(wire_list) - 1 and wire in gate.targets:
                self._render_strs["top_frame"][wire] += top_frame
                self._render_strs["mid_frame"][wire] += mid_connect
                self._render_strs["bot_frame"][wire] += mid_frame
            else:
                self._render_strs["top_frame"][wire] += mid_frame
                self._render_strs["mid_frame"][wire] += mid_frame
                self._render_strs["bot_frame"][wire] += mid_frame

    def _update_control_multiq(
        self,
        gate: Gate,
        wire_list_control: List[int],
        xskip: int,
        width: int,
        is_top_closer: bool,
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

        bar_conn = " " * (width // 2) + "│" + " " * (width // 2 - 1)
        node_conn = "─" * (width // 2) + "▇" + "─" * (width // 2 - 1)
        self._adjust_layer(wire_list_control, xskip)

        for wire in wire_list_control:
            if wire not in gate.targets:
                if wire in gate.controls:
                    if (
                        wire == wire_list_control[0]
                        or wire == wire_list_control[-1]
                    ):
                        self._render_strs["mid_frame"][wire] += node_conn
                        self._render_strs["top_frame"][wire] += (
                            bar_conn if is_top_closer else " " * len(bar_conn)
                        )
                        self._render_strs["bot_frame"][wire] += (
                            bar_conn
                            if not is_top_closer
                            else " " * len(bar_conn)
                        )
                    else:
                        self._render_strs["top_frame"][wire] += bar_conn
                        self._render_strs["mid_frame"][wire] += node_conn
                        self._render_strs["bot_frame"][wire] += bar_conn
                else:
                    self._render_strs["top_frame"][wire] += bar_conn
                    self._render_strs["mid_frame"][wire] += bar_conn
                    self._render_strs["bot_frame"][wire] += bar_conn

    def _update_swap_gate(self, gate: Gate):
        """
        Update the render strings for the SWAP gate
        """

        wire_list = list(range(min(gate.targets), max(gate.targets) + 1))
        layer = max(len(self._layer_list[i]) for i in wire_list)
        xskip = self.get_xskip(wire_list, layer)
        self._adjust_layer(wire_list, xskip)

        width = 4 * self.gate_pad + 1
        cross_conn = "─" * (width // 2) + "╳" + "─" * (width // 2)
        bar_conn = " " * (width // 2) + "│" + " " * (width // 2)
        mid_bar_conn = "─" * (width // 2) + "│" + "─" * (width // 2)
        self.manage_layers(wire_list, layer, xskip, width)

        for wire in wire_list:
            if wire == wire_list[-1]:
                self._render_strs["top_frame"][wire] += " " * len(bar_conn)
                self._render_strs["mid_frame"][wire] += cross_conn
                self._render_strs["bot_frame"][wire] += bar_conn
            elif wire == wire_list[0]:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["mid_frame"][wire] += cross_conn
                self._render_strs["bot_frame"][wire] += " " * len(bar_conn)
            else:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["mid_frame"][wire] += mid_bar_conn
                self._render_strs["bot_frame"][wire] += bar_conn

    def layout(self):
        """
        Layout the circuit
        """
        self._add_wire_labels()

        for gate in self._qc.gates:
            wire_list, layer, xskip, parts, width = [], 0, 0, [], 0

            # generate the parts, width and wire_list for the gate
            if isinstance(gate, Measurement):
                wire_list = list(range(gate.targets[0] + 1)) + list(
                    range(
                        gate.classical_store + self._qwires,
                        self._qwires + self._cwires,
                    )
                )
                parts, width = self._draw_measurement_gate(gate)
            elif len(gate.targets) == 1 and gate.controls is None:
                wire_list = gate.targets
                parts, width = self._draw_singleq_gate(gate.name)
            elif gate.name == "SWAP":
                self._update_swap_gate(gate)
                continue
            else:
                merged_wire = sorted(gate.targets + (gate.controls or []))
                wire_list = list(range(merged_wire[0], merged_wire[-1] + 1))
                parts, width = self._draw_multiq_gate(gate)

            # update the render strings for the gate
            layer = max(len(self._layer_list[i]) for i in wire_list)
            xskip = self.get_xskip(wire_list, layer)
            self.manage_layers(wire_list, layer, xskip, width)

            if isinstance(gate, Measurement):
                self._adjust_layer(wire_list, xskip)
                self._update_singleq(gate.targets, parts)
                self._update_cbridge(gate, wire_list, xskip, width)
            elif len(gate.targets) == 1 and gate.controls is None:
                self._update_singleq(wire_list, parts)
            else:
                sorted_targets = sorted(gate.targets)
                self._update_target_multiq(
                    gate,
                    list(range(sorted_targets[0], sorted_targets[-1] + 1)),
                    xskip,
                    parts,
                )

                if gate.controls:
                    sorted_controls = sorted(gate.controls)
                    is_top_closer = wire_list[0] >= sorted_controls[0]
                    closest_pos = (
                        wire_list[0] if not is_top_closer else wire_list[-1]
                    )
                    self._update_control_multiq(
                        gate,
                        list(
                            range(
                                min(sorted_controls[0], closest_pos),
                                max(sorted_controls[-1], closest_pos) + 1,
                            )
                        ),
                        xskip,
                        width,
                        is_top_closer,
                    )

        self._extend_line()
        self.print_circuit()

    def print_circuit(self):
        """
        Print the circuit
        """

        for i in range(self._qwires - 1, -1, -1):
            print(self._render_strs["top_frame"][i])
            print(self._render_strs["mid_frame"][i])
            print(self._render_strs["bot_frame"][i])

        for i in range(self._qwires + self._cwires - 1, self._qwires - 1, -1):
            print(self._render_strs["top_frame"][i])
            print(self._render_strs["mid_frame"][i])
            print(self._render_strs["bot_frame"][i])
