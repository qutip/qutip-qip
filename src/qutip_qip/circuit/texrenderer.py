from functools import partialmethod

from ..operations import Gate
from . import circuit_latex as _latex

__all__ = ["TeXRenderer"]


class TeXRenderer:
    """
    Class to render the circuit in latex format.
    """

    def __init__(self, qc):

        self.qc = qc
        self.N = qc.N
        self.num_cbits = qc.num_cbits
        self.gates = qc.gates
        self.input_states = qc.input_states
        self.reverse_states = qc.reverse_states

        self._latex_template = r"""
        \documentclass[border=3pt]{standalone}
        \usepackage[braket]{qcircuit}
        \begin{document}
        \Qcircuit @C=1cm @R=1cm {
        %s}
        \end{document}
        """

        if "png" in _latex.CONVERTERS:
            self._repr_png_ = self.raw_img

        if "svg" in _latex.CONVERTERS:
            self._repr_svg_ = partialmethod(
                TeXRenderer.raw_img, file_type="svg", dpi=None
            )

    def _gate_label(self, gate):
        gate_label = gate.latex_str
        if gate.arg_label is not None:
            return r"%s(%s)" % (gate_label, gate.arg_label)
        return r"%s" % gate_label

    def latex_code(self):
        """
        Generate the latex code for the circuit.

        Returns
        -------
        code: str
            The latex code for the circuit.
        """

        rows = []

        ops = self.gates
        col = []
        for op in ops:
            if isinstance(op, Gate):
                gate = op
                col = []
                _swap_processing = False
                for n in range(self.N + self.num_cbits):
                    if gate.targets and n in gate.targets:
                        if len(gate.targets) > 1:
                            if gate.name == "SWAP":
                                if _swap_processing:
                                    col.append(r" \qswap \qw")
                                    continue
                                distance = abs(
                                    gate.targets[1] - gate.targets[0]
                                )
                                if self.reverse_states:
                                    distance = -distance
                                col.append(r" \qswap \qwx[%d] \qw" % distance)
                                _swap_processing = True

                            elif (
                                self.reverse_states and n == max(gate.targets)
                            ) or (
                                not self.reverse_states
                                and n == min(gate.targets)
                            ):
                                col.append(
                                    r" \multigate{%d}{%s} "
                                    % (
                                        len(gate.targets) - 1,
                                        self._gate_label(gate),
                                    )
                                )
                            else:
                                col.append(
                                    r" \ghost{%s} " % (self._gate_label(gate))
                                )

                        elif gate.name == "CNOT":
                            col.append(r" \targ ")
                        elif gate.name == "CY":
                            col.append(r" \targ ")
                        elif gate.name == "CZ":
                            col.append(r" \targ ")
                        elif gate.name == "CS":
                            col.append(r" \targ ")
                        elif gate.name == "CT":
                            col.append(r" \targ ")
                        elif gate.name == "TOFFOLI":
                            col.append(r" \targ ")
                        else:
                            col.append(r" \gate{%s} " % self._gate_label(gate))

                    elif gate.controls and n in gate.controls:
                        control_tag = (-1 if self.reverse_states else 1) * (
                            gate.targets[0] - n
                        )
                        col.append(r" \ctrl{%d} " % control_tag)

                    elif (
                        gate.classical_controls
                        and (n - self.N) in gate.classical_controls
                    ):
                        control_tag = (-1 if self.reverse_states else 1) * (
                            gate.targets[0] - n
                        )
                        col.append(r" \ctrl{%d} " % control_tag)

                    elif not gate.controls and not gate.targets:
                        # global gate
                        if (self.reverse_states and n == self.N - 1) or (
                            not self.reverse_states and n == 0
                        ):
                            col.append(
                                r" \multigate{%d}{%s} "
                                % (
                                    self.N - 1,
                                    self._gate_label(gate),
                                )
                            )
                        else:
                            col.append(
                                r" \ghost{%s} " % (self._gate_label(gate))
                            )
                    else:
                        col.append(r" \qw ")

            else:
                measurement = op
                col = []
                for n in range(self.N + self.num_cbits):
                    if n in measurement.targets:
                        col.append(r" \meter")
                    elif (n - self.N) == measurement.classical_store:
                        sgn = 1 if self.reverse_states else -1
                        store_tag = sgn * (n - measurement.targets[0])
                        col.append(r" \qw \cwx[%d] " % store_tag)
                    else:
                        col.append(r" \qw ")

            col.append(r" \qw ")
            rows.append(col)

        input_states_quantum = [
            r"\lstick{\ket{" + x + "}}" if x is not None else ""
            for x in self.input_states[: self.N]
        ]
        input_states_classical = [
            r"\lstick{" + x + "}" if x is not None else ""
            for x in self.input_states[self.N :]
        ]
        input_states = input_states_quantum + input_states_classical

        code = ""
        n_iter = (
            reversed(range(self.N + self.num_cbits))
            if self.reverse_states
            else range(self.N + self.num_cbits)
        )
        for n in n_iter:
            code += r" & %s" % input_states[n]
            for m in range(len(ops)):
                code += r" & %s" % rows[m][n]
            code += r" & \qw \\ " + "\n"

        return self._latex_template % code

    def raw_img(self, file_type="png", dpi=100):
        return _latex.image_from_latex(self.latex_code(), file_type, dpi)