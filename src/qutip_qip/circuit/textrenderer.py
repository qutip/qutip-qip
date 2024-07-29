from typing import Union, Optional, List, Dict
from dataclasses import dataclass

import numpy as np

from ..operations import Gate, Measurement
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

        self._render_strs = {
            "top": ["  "] * self._qwires,
            "mid": ["──"] * self._qwires,
            "bot": ["  "] * self._qwires,
        }

        # add wire labels
        for i in range(self._qwires):
            self._render_strs["mid"][i] = (
                f"q{i} :" + self._render_strs["mid"][i]
            )
            self._render_strs["top"][i] = "    " + self._render_strs["top"][i]
            self._render_strs["bot"][i] = "    " + self._render_strs["bot"][i]

        self._layer_list = {
            i: [len(self._render_strs["mid"][i])] for i in range(self._qwires)
        }

    def extend_line(self):
        """
        Extend the wires on the circuit
        """

        length_wire = [sum(self._layer_list[i]) for i in range(self._qwires)]
        max_length = max(length_wire)

        for i in range(self._qwires):
            if length_wire[i] < max_length:
                diff = max_length - length_wire[i]
                self._render_strs["top"][i] += " " * diff
                self._render_strs["mid"][i] += "─" * diff
                self._render_strs["bot"][i] += " " * diff

    def get_xskip(self, wire_list, layer):

        xskip = []
        for wire in wire_list:
            xskip.append(sum(self._layer_list[wire][:layer]))

        return max(xskip)

    def manage_layers(
        self, wire_list, layer, xskip, gate_width, gate_margin=0
    ):

        for wire in wire_list:
            if len(self._layer_list[wire]) > layer:
                if (
                    self._layer_list[wire][layer]
                    < gate_width + gate_margin * 2
                ):
                    self._layer_list[wire][layer] = (
                        gate_width + gate_margin * 2
                    )
            else:
                temp = xskip - sum(self._layer_list[wire]) if xskip != 0 else 0
                self._layer_list[wire].append(
                    temp + gate_width + gate_margin * 2
                )

    def draw_singleq_gate(self, gate):
        """
        Draw a single qubit gate
        """

        gap_len = 1
        s = "─" * (gap_len * 2 + len(gate.name))
        gap = " " * gap_len

        top = f" ┌{s}┐ "
        mid = f"─┤{gap}{gate.name}{gap}├─"
        bot = f" └{s}┘ "

        return top, mid, bot, len(top)

    # TODO: sort this as well
    def draw_multiq_gate(self, gate):
        """
        Draw a multi qubit gate
        """

        gap_len = 1
        s = "─" * (len(gate.name) + 2)
        gap = " " * gap_len
        top = f" ┌{s}┐ "
        bot = f" └{s}┘ "

        mid = f" |{gap}{" "*len(gate.name)}{gap}| "
        connect = f"─┤{gap}{" "*len(gate.name)}{gap}├─"
        connect_label = f"─┤{gap}{gate.name}{gap}├─"

        # if control node is in wire above top wire
        # place "┴" in the center of top
        # if control node is in wire below bot wire
        # place "┬" in the center of bot

        # TODO fix: what if the control nodes are on both sides
        if gate.controls is not None:
            control_wire = gate.controls[0]
            if control_wire < gate.targets[0]:
                top = top[: len(top) // 2] + "┴" + top[len(top) // 2 + 1 :]
            else:
                bot = bot[: len(bot) // 2] + "┬" + bot[len(bot) // 2 + 1 :]

        return top, mid, connect, connect_label, bot, len(top)

    def print_circuit(self):
        """
        Print the circuit
        """

        for i in range(self._qwires):
            print(self._render_strs["top"][i])
            print(self._render_strs["mid"][i])
            print(self._render_strs["bot"][i])

    def adjust_layer(self, xskip, wire_list):
        """
        Adjust the layer
        """

        for wire in wire_list:
            self._render_strs["top"][wire] += " " * (
                xskip - len(self._render_strs["top"][wire])
            )
            self._render_strs["mid"][wire] += "─" * (
                xskip - len(self._render_strs["mid"][wire])
            )
            self._render_strs["bot"][wire] += " " * (
                xskip - len(self._render_strs["bot"][wire])
            )

    def layout(self):
        """
        Layout the circuit
        """

        for gate in self._qc.gates:

            if gate is type(Measurement):
                raise NotImplementedError(
                    "Measurement gates are not supported yet"
                )

            if len(gate.targets) == 1 and gate.controls is None:

                wire_list = gate.targets
                layer = max([len(self._layer_list[i]) for i in gate.targets])
                xskip = self.get_xskip(wire_list, layer)

                top, mid, bot, width = self.draw_singleq_gate(gate)
                self.manage_layers(wire_list, layer, xskip, width)

                for i, wire in enumerate(wire_list):
                    self._render_strs["top"][wire] += top
                    self._render_strs["mid"][wire] += mid
                    self._render_strs["bot"][wire] += bot

            else:

                # TODO sort this out !!!
                merged_wire = gate.targets.copy()
                if gate.controls is not None:
                    merged_wire += gate.controls

                merged_wire = sorted(merged_wire)
                wire_list = list(range(merged_wire[0], merged_wire[-1] + 1))
                layer = max([len(self._layer_list[i]) for i in wire_list])
                xskip = self.get_xskip(wire_list, layer)
                top, mid, connect, connect_label, bot, width = (
                    self.draw_multiq_gate(gate)
                )
                self.manage_layers(wire_list, layer, xskip, width)

                # loop thorugh target
                sorted_targets = sorted(gate.targets)
                wire_list_target = list(
                    range(sorted_targets[0], sorted_targets[-1] + 1)
                )

                self.adjust_layer(xskip, wire_list_target)
                # main logic
                for i, wire in enumerate(wire_list_target):

                    if len(gate.targets) == 1:
                        self._render_strs["top"][wire] += top
                        self._render_strs["mid"][wire] += connect_label
                        self._render_strs["bot"][wire] += bot
                    elif i == 0 and wire in gate.targets:
                        self._render_strs["top"][wire] += top
                        self._render_strs["mid"][wire] += connect
                        self._render_strs["bot"][wire] += mid
                    elif (
                        i == len(wire_list_target) - 1 and wire in gate.targets
                    ):
                        self._render_strs["top"][wire] += mid
                        self._render_strs["mid"][wire] += connect_label
                        self._render_strs["bot"][wire] += bot
                    else:
                        self._render_strs["top"][wire] += mid
                        self._render_strs["mid"][wire] += mid
                        self._render_strs["bot"][wire] += mid

                # find if top or bot wire is closer to control wire
                if gate.controls is not None:

                    sorted_controls = sorted(gate.controls)
                    if wire_list_target[0] < sorted_controls[0]:
                        closest_pos = wire_list_target[0]
                        is_top_closer = True
                    else:
                        closest_pos = wire_list_target[-1]
                        is_top_closer = False

                    # loop through control
                    wire_list_control = list(
                        range(
                            min(sorted_controls[0], closest_pos),
                            max(sorted_controls[-1], closest_pos) + 1,
                        )
                    )

                    self.adjust_layer(xskip, wire_list_control)
                    for i, wire in enumerate(wire_list_control):

                        if wire not in gate.targets:
                            if wire in gate.controls:
                                
                                self._render_strs["mid"][wire] += (
                                    "─" * (width // 2)
                                    + "▇"
                                    + "─" * (width // 2)
                                )
                                self._render_strs["top"][wire] += (
                                    " " * (width // 2)
                                    + "|"
                                    + " " * (width // 2)
                                    if is_top_closer
                                    else " " * width
                                )
                                self._render_strs["bot"][wire] += (
                                    " " * (width // 2)
                                    + "|"
                                    + " " * (width // 2)
                                    if not is_top_closer
                                    else " " * width
                                )
                            else:
                                self._render_strs["top"][wire] += (
                                    " " * (width // 2)
                                    + "|"
                                    + " " * (width // 2)
                                )
                                self._render_strs["mid"][wire] += (
                                    " " * (width // 2)
                                    + "|"
                                    + " " * (width // 2)
                                )
                                self._render_strs["bot"][wire] += (
                                    " " * (width // 2)
                                    + "|"
                                    + " " * (width // 2)
                                )

        self.extend_line()
        self.print_circuit()
