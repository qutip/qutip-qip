from copy import deepcopy

from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate


_GATE_NAME_TO_QASM_NAME: dict[str: str] = {
    "QASMU": "U",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "SNOT": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "CRZ": "crz",
    "CNOT": "cx",
    "TOFFOLI": "ccx",
}

def print_qasm(qc: QubitCircuit):
    """
    Print QASM output of circuit object.

    Parameters
    ----------
    qc : :class:`.QubitCircuit`
        circuit object to produce QASM output for.
    """

    qasm_out = QasmOutput("2.0")
    lines = qasm_out._qasm_output(qc)
    for line in lines:
        print(line)


def circuit_to_qasm_str(qc: QubitCircuit) -> str:
    """
    Return QASM output of circuit object as string

    Parameters
    ----------
    qc : :class:`.QubitCircuit`
        circuit object to produce QASM output for.

    Returns
    -------
    output: str
        string corresponding to QASM output.
    """

    qasm_out = QasmOutput("2.0")
    lines = qasm_out._qasm_output(qc)
    output = ""
    for line in lines:
        output += line + "\n"
    return output


def save_qasm(qc: QubitCircuit, file_loc: str):
    """
    Save QASM output of circuit object to file.

    Parameters
    ----------
    qc : :class:`.QubitCircuit`
        circuit object to produce QASM output for.

    file_loc : str
        File path where the qasm output needs to be saved.
    """

    qasm_out = QasmOutput("2.0")
    lines = qasm_out._qasm_output(qc)
    with open(file_loc, "w") as f:
        for line in lines:
            f.write("{}\n".format(line))


class QasmOutput:
    """
    Class for QASM export.

    Parameters
    ----------
    version: str, optional
        OpenQASM version, currently must be "2.0" necessarily.
    """

    def __init__(self, version: str = "2.0"):
        self.version = version
        self.lines = []
        self.gate_name_map = deepcopy(_GATE_NAME_TO_QASM_NAME)

    def output(self, line: str = "", n: int = 0):
        """
        Pipe QASM output string to QasmOutput's lines variable.

        Parameters
        ----------
        line: str, optional
            string to be appended to QASM output.
        n: int, optional
            number of blank lines to be appended to QASM output.
        """

        if line:
            self.lines.append(line)
        self.lines = self.lines + [""] * n

    def _flush(self):
        """
        Resets QasmOutput variables.
        """

        self.lines = []
        self.gate_name_map = deepcopy(_GATE_NAME_TO_QASM_NAME)

    def _qasm_str(self, q_name, q_controls, q_targets, q_args):
        """
        Returns QASM string for gate definition or gate application given
        name, registers, arguments.
        """

        if not q_controls:
            q_controls = []
        q_regs = q_controls + q_targets

        if isinstance(q_targets[0], int):
            q_regs = ",".join(["q[{}]".format(reg) for reg in q_regs])
        else:
            q_regs = ",".join(q_regs)

        if q_args:
            if isinstance(q_args, list):
                q_args = ",".join([str(arg) for arg in q_args])
            return "{}({}) {};".format(q_name, q_args, q_regs)
        else:
            return "{} {};".format(q_name, q_regs)

    def _qasm_defn_from_resolved(
        self,
        curr_gate: Gate,
        gates_lst: list[Gate]
    ):
        """
        Resolve QASM definition of QuTiP gate in terms of component gates.

        Parameters
        ----------
        curr_gate: :class:`~.operations.Gate`
            QuTiP gate which needs to be resolved into component gates.
        gates_lst: list of :class:`~.operations.Gate`
            list of gate that constitute QASM definition of self.
        """

        forbidden_gates = ["GLOBALPHASE", "PHASEGATE"]
        reg_map = ["a", "b", "c"]

        q_controls = None
        if curr_gate.controls:
            q_controls = [reg_map[i] for i in curr_gate.controls]
        q_targets = [reg_map[i] for i in curr_gate.targets]
        arg_name = None
        if curr_gate.arg_value:
            arg_name = "theta"

        self.output(
            "gate {} {{".format(
                self._qasm_str(
                    curr_gate.name.lower(), q_controls, q_targets, arg_name
                )[:-1]
            )
        )

        for gate in gates_lst:
            if gate.name in self.gate_name_map:
                gate.targets = [reg_map[i] for i in gate.targets]
                if gate.controls:
                    gate.controls = [reg_map[i] for i in gate.controls]
                self.output(
                    self._qasm_str(
                        self.gate_name_map[gate.name],
                        gate.controls,
                        gate.targets,
                        gate.arg_value,
                    )
                )
            elif gate.name in forbidden_gates:
                continue
            else:
                raise ValueError(
                    (
                        "The given resolved gate {} cannot be defined"
                        " in QASM format"
                    ).format(curr_gate.name)
                )
        self.output("}")

    def _qasm_defn_resolve(self, gate: Gate):
        """
        Resolve QASM definition of QuTiP gate if possible.

        Parameters
        ----------
        gate: :class:`~.operations.Gate`
            QuTiP gate which needs to be resolved into component gates.

        """

        qc = QubitCircuit(3)
        gates_lst = []
        if gate.name == "CSIGN":
            qc._gate_CSIGN(gate, gates_lst)
        else:
            err_msg = "No definition specified for {} gate".format(gate.name)
            raise NotImplementedError(err_msg)

        self._qasm_defn_from_resolved(gate, gates_lst)
        self.gate_name_map[gate.name] = gate.name.lower()

    def _qasm_defns(self, gate: Gate):
        """
        Define QASM gates for QuTiP gates that do not have QASM counterparts.

        Parameters
        ----------
        gate: :class:`~.operations.Gate`
            QuTiP gate which needs to be defined in QASM format.
        """

        if gate.name == "CRY":
            gate_def = "gate cry(theta) a,b { cu3(theta,0,0) a,b; }"
        elif gate.name == "CRX":
            gate_def = "gate crx(theta) a,b { cu3(theta,-pi/2,pi/2) a,b; }"
        elif gate.name == "SQRTNOT":
            gate_def = "gate sqrtnot a {h a; u1(-pi/2) a; h a; }"
        elif gate.name == "CS":
            gate_def = "gate cs a,b { cu1(pi/2) a,b; }"
        elif gate.name == "CT":
            gate_def = "gate ct a,b { cu1(pi/4) a,b; }"
        elif gate.name == "SWAP":
            gate_def = "gate swap a,b { cx a,b; cx b,a; cx a,b; }"
        else:
            self._qasm_defn_resolve(gate)
            return

        self.output("// QuTiP definition for gate {}".format(gate.name))
        self.output(gate_def)
        self.gate_name_map[gate.name] = gate.name.lower()

    def qasm_name(self, gate_name: str) -> (str | None):
        """
        Return QASM gate name for corresponding QuTiP gate.

        Parameters
        ----------
        gate_name: str
            QuTiP gate name.
        """

        if gate_name in self.gate_name_map:
            return self.gate_name_map[gate_name]
        else:
            return None

    def is_defined(self, gate_name: str) -> bool:
        """
        Check if QASM gate definition exists for QuTiP gate.

        Parameters
        ----------
        gate_name: str
            QuTiP gate name.
        """

        return gate_name in self.gate_name_map

    def _qasm_output(self, qc: QubitCircuit):
        """
        QASM output handler.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            circuit object to produce QASM output for.
        """

        self._flush()

        self.output("// QASM 2.0 file generated by QuTiP", 1)

        if self.version == "2.0":
            self.output("OPENQASM 2.0;")
        else:
            raise NotImplementedError(
                "QASM: Only OpenQASM 2.0 \
                                      is currently supported."
            )

        self.output('include "qelib1.inc";', 1)

        qc._to_qasm(self)

        return self.lines
