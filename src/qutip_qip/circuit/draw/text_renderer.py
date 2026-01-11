"""
Module for rendering a quantum circuit in text format.
"""

from math import ceil

from qutip_qip.circuit import QubitCircuit
from qutip_qip.circuit.draw import BaseRenderer, StyleConfig
from qutip_qip.operations import Gate, Measurement


class TextRenderer(BaseRenderer):
    """
    A class to render a quantum circuit in text format.
    """

    def __init__(self, qc: QubitCircuit, **style) -> None:
        # user defined style
        style = {} if style is None else style
        style["gate_margin"] = 0
        self.style = StyleConfig(**style)

        super().__init__(self.style)
        self._qc = qc
        self._qwires = qc.N
        self._cwires = qc.num_cbits
        self._layer_list = [[] for _ in range(self._qwires + self._cwires)]

        self._render_strs = {
            "top_frame": ["  "] * (self._qwires + self._cwires),
            "mid_frame": ["──"] * self._qwires + ["══"] * self._cwires,
            "bot_frame": ["  "] * (self._qwires + self._cwires),
        }

    def _adjust_layer_pad(self, wire_list: list[int], xskip: int) -> None:
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

    def _add_wire_labels(self) -> None:
        """
        Add wire labels to the circuit
        """

        if self.style.wire_label is None:
            default_labels = [f"q{i}" for i in range(self._qwires)] + [
                f"c{i}" for i in range(self._cwires)
            ]
        else:
            default_labels = (
                self.style.wire_label[self._cwires :]
                + self.style.wire_label[: self._cwires]
            )

        max_label_len = max([len(label) for label in default_labels])
        for i, label in enumerate(default_labels):
            self._render_strs["mid_frame"][i] = (
                f" {label} "
                + " " * (max_label_len - len(label))
                + ":"
                + self._render_strs["mid_frame"][i]
            )

            update_len = len(self._render_strs["mid_frame"][i])
            self._render_strs["top_frame"][i] = " " * update_len
            self._render_strs["bot_frame"][i] = " " * update_len

            self._layer_list[i].append(update_len)

    def _draw_singleq_gate(
        self, gate_name: str
    ) -> tuple[tuple[str, str, str], int]:
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

        lid_seg = "─" * (ceil(self.style.gate_pad) * 2 + len(gate_name))
        pad = " " * ceil(self.style.gate_pad)

        top_frame = f" ┌{lid_seg}┐ "
        mid_frame = f"─┤{pad}{gate_name}{pad}├─"
        bot_frame = f" └{lid_seg}┘ "

        # check for equal part lengths
        assert len(top_frame) == len(mid_frame) == len(bot_frame)

        return (top_frame, mid_frame, bot_frame), len(top_frame)

    def _draw_multiq_gate(
        self, gate: Gate, gate_text: str
    ) -> tuple[tuple[str, str, str, str, str], int]:
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

        lid_seg = "─" * (ceil(self.style.gate_pad) * 2 + len(gate_text))
        pad = " " * ceil(self.style.gate_pad)

        top_frame = f" ┌{lid_seg}┐ "
        bot_frame = f" └{lid_seg}┘ "
        mid_frame = f" │{pad}{' ' * len(gate_text)}{pad}│ "
        mid_connect = f"─┤{pad}{' ' * len(gate_text)}{pad}├─"
        mid_connect_label = f"─┤{pad}{gate_text}{pad}├─"
        mid_index = len(bot_frame) // 2

        sorted_targets = sorted(gate.targets)
        # Adjust top_frame or bottom if there is a control wire
        if gate.controls:
            sorted_controls = sorted(gate.controls)
            top_frame = (
                (top_frame[:mid_index] + "┴" + top_frame[mid_index + 1 :])
                if sorted_controls[-1] > sorted_targets[0]
                else top_frame
            )
            bot_frame = (
                (bot_frame[:mid_index] + "┬" + bot_frame[mid_index + 1 :])
                if sorted_controls[0] < sorted_targets[-1]
                else bot_frame
            )

        # Adjust the frames for classical_controls
        if gate.classical_controls is not None:
            for c in gate.classical_controls:
                if c + self._qwires > sorted_targets[0]:
                    bot_frame = (
                        bot_frame[:mid_index]
                        + "╥"
                        + bot_frame[mid_index + 1 :]
                    )
                else:
                    top_frame = (
                        top_frame[:mid_index]
                        + "╨"
                        + top_frame[mid_index + 1 :]
                    )

        # check for equal part lengths
        assert (
            len(top_frame)
            == len(mid_frame)
            == len(bot_frame)
            == len(mid_connect)
            == len(mid_connect_label)
        )

        return (
            top_frame,
            mid_frame,
            mid_connect,
            mid_connect_label,
            bot_frame,
        ), len(top_frame)

    def _draw_measurement_gate(
        self, measurement: Measurement
    ) -> tuple[tuple[str, str, str], int]:
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
        self,
        gate: Gate,
        wire_list: list[int],
        width: int,
        is_arrow: bool = True,
    ) -> None:
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

        bar_conn = " " * (width // 2) + "║" + " " * (width // 2 - 1)
        mid_bar_conn = "─" * (width // 2) + "║" + "─" * (width // 2 - 1)
        mid_bar_classical_conn = (
            "═" * (width // 2) + "║" + "═" * (width // 2 - 1)
        )
        classical_conn = (
            "═" * (width // 2)
            + ("╩" if is_arrow else "█")
            + "═" * (width // 2 - 1)
        )

        classical_wire = list(
            map(
                lambda x: x + self._qwires,
                (
                    gate.classical_controls
                    if isinstance(gate, Gate)
                    else [gate.classical_store]
                ),
            )
        )
        classical_wire.sort()

        for wire in wire_list:
            if wire == gate.targets[0]:
                continue
            if wire in classical_wire:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["mid_frame"][wire] += classical_conn
                self._render_strs["bot_frame"][wire] += (
                    " " * len(bar_conn)
                    if wire < classical_wire[-1] or len(classical_wire) == 1
                    else bar_conn
                )
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

    def _update_singleq(self, wire_list, parts) -> None:
        """
        Update the render strings for single qubit gates
        """
        top_frame, mid_frame, bot_frame = parts
        for wire in wire_list:
            self._render_strs["top_frame"][wire] += top_frame
            self._render_strs["mid_frame"][wire] += mid_frame
            self._render_strs["bot_frame"][wire] += bot_frame

    def _update_target_multiq(
        self, gate: Gate, wire_list: list[int], parts: list[str]
    ) -> None:
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

    def _update_qbridge(
        self,
        gate: Gate,
        wire_list_control: list[int],
        width: int,
        is_top: bool,
    ) -> None:
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
        mid_bar_conn = "─" * (width // 2) + "│" + "─" * (width // 2 - 1)
        node_conn = "─" * (width // 2) + "█" + "─" * (width // 2 - 1)

        for wire in wire_list_control:
            if wire not in gate.targets:
                if wire in gate.controls:
                    # check if the control wire is the first or last control wire.
                    # used in cases of multiple control wires
                    if (
                        wire == wire_list_control[0]
                        or wire == wire_list_control[-1]
                    ):
                        self._render_strs["mid_frame"][wire] += node_conn
                        self._render_strs["top_frame"][wire] += (
                            bar_conn if not is_top else " " * len(bar_conn)
                        )
                        self._render_strs["bot_frame"][wire] += (
                            bar_conn if is_top else " " * len(bar_conn)
                        )
                    else:
                        self._render_strs["top_frame"][wire] += bar_conn
                        self._render_strs["mid_frame"][wire] += node_conn
                        self._render_strs["bot_frame"][wire] += bar_conn
                else:
                    self._render_strs["top_frame"][wire] += bar_conn
                    self._render_strs["mid_frame"][wire] += mid_bar_conn
                    self._render_strs["bot_frame"][wire] += bar_conn

    def _update_swap_gate(self, wire_list: list[int]) -> None:
        """
        Update the render strings for the SWAP gate
        """

        width = 4 * ceil(self.style.gate_pad) + 1
        cross_conn = "─" * (width // 2) + "╳" + "─" * (width // 2)
        bar_conn = " " * (width // 2) + "│" + " " * (width // 2)
        mid_bar_conn = "─" * (width // 2) + "│" + "─" * (width // 2)

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

    def layout(self) -> None:
        """
        Layout the circuit
        """
        self._add_wire_labels()

        for gate in self._qc.gates:
            if isinstance(gate, Gate):
                gate_text = (
                    gate.arg_label if gate.arg_label is not None else gate.name
                )

            # generate the parts, width and wire_list for the gates
            if isinstance(gate, Measurement):
                wire_list = list(range(gate.targets[0] + 1)) + list(
                    range(
                        gate.classical_store + self._qwires,
                        self._qwires + self._cwires,
                    )
                )
                parts, width = self._draw_measurement_gate(gate)
            elif gate.name == "SWAP":
                wire_list = list(
                    range(min(gate.targets), max(gate.targets) + 1)
                )
                width = 4 * ceil(self.style.gate_pad) + 1
            else:
                sorted_targets = sorted(gate.targets)
                merged_wire = sorted_targets + (gate.controls or [])
                if gate.classical_controls is not None:
                    c_control = sorted(gate.classical_controls)
                    merged_wire += list(range(sorted_targets[0] + 1))
                    merged_wire.sort()
                    wire_list = list(
                        range(merged_wire[0], merged_wire[-1] + 1)
                    )
                    wire_list += list(
                        range(
                            c_control[0] + self._qwires,
                            self._qwires + self._cwires,
                        )
                    )
                else:
                    merged_wire.sort()
                    wire_list = list(
                        range(merged_wire[0], merged_wire[-1] + 1)
                    )
                parts, width = self._draw_multiq_gate(gate, gate_text)

            # update the render strings for the gate
            layer = max(len(self._layer_list[i]) for i in wire_list)
            xskip = self._get_xskip(wire_list, layer)
            self._adjust_layer_pad(wire_list, xskip)
            self._manage_layers(width, wire_list, layer, xskip)

            if isinstance(gate, Measurement):
                self._update_singleq(gate.targets, parts)
                self._update_cbridge(gate, wire_list, width)
            elif gate.name == "SWAP":
                self._update_swap_gate(wire_list)
            else:
                self._update_target_multiq(
                    gate,
                    list(range(sorted_targets[0], sorted_targets[-1] + 1)),
                    parts,
                )

                if gate.controls:
                    sorted_controls = sorted(gate.controls)

                    # check if there is control wire above the gate top
                    is_top = sorted_controls[-1] > sorted_targets[0]
                    is_bot = sorted_controls[0] < sorted_targets[-1]

                    if is_top:
                        self._update_qbridge(
                            gate,
                            list(
                                range(
                                    sorted_targets[-1],
                                    sorted_controls[-1] + 1,
                                )
                            ),
                            width,
                            is_top,
                        )

                    if is_bot:
                        self._update_qbridge(
                            gate,
                            list(
                                range(
                                    sorted_controls[0],
                                    sorted_targets[-1] + 1,
                                )
                            ),
                            width,
                            not is_bot,
                        )

                if gate.classical_controls is not None:
                    self._update_cbridge(
                        gate,
                        list(range(sorted_targets[0] + 1))
                        + list(
                            range(
                                gate.classical_controls[0] + self._qwires,
                                self._qwires + self._cwires,
                            )
                        ),
                        width,
                        is_arrow=False,
                    )

        max_layer_len = max(sum(layer) for layer in self._layer_list)
        self._adjust_layer_pad(
            list(range(self._qwires + self._cwires)),
            max_layer_len + self.style.end_wire_ext,
        )
        self.print_circuit()

    def print_circuit(self) -> None:
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

    def save(self, filename: str) -> None:
        """
        Save the circuit to a file

        Parameters
        ----------
        filename : str
            The name of the file to save the circuit to.
        """

        if not filename.endswith(".txt"):
            filename += ".txt"

        with open(filename, "w", encoding="utf-8") as file:
            for i in range(self._qwires - 1, -1, -1):
                file.write(self._render_strs["top_frame"][i] + "\n")
                file.write(self._render_strs["mid_frame"][i] + "\n")
                file.write(self._render_strs["bot_frame"][i] + "\n")

            for i in range(
                self._qwires + self._cwires - 1, self._qwires - 1, -1
            ):
                file.write(self._render_strs["top_frame"][i] + "\n")
                file.write(self._render_strs["mid_frame"][i] + "\n")
                file.write(self._render_strs["bot_frame"][i] + "\n")
