from collections.abc import Iterable
from copy import deepcopy
from typing import Self, Type

import qutip_qip.operations.gates as gates
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate

_GATE_NAME_TO_QASM_NAME = {
    "QASMU": "U",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "SNOT": "h",
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "CRZ": "crz",
    "CX": "cx",
    "TOFFOLI": "ccx",
}


def print_qasm(qc: QubitCircuit) -> None:
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


def save_qasm(qc: QubitCircuit, file_loc: str) -> None:
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
            f.write(f"{line}\n")


class QasmOutput:
    """
    Class for QASM export.

    Parameters
    ----------
    version: str, optional
        OpenQASM version, currently must be "2.0" necessarily.
    """

    def __init__(self: Self, version: str = "2.0") -> None:
        self.version = version
        self.lines = []
        self.gate_name_map = deepcopy(_GATE_NAME_TO_QASM_NAME)

    def output(self: Self, line: str = "", n: int = 0) -> None:
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

    def _flush(self: Self):
        """
        Resets QasmOutput variables.
        """

        self.lines = []
        self.gate_name_map = deepcopy(_GATE_NAME_TO_QASM_NAME)

    def _qasm_str(
        self,
        q_name: str,
        q_targets: list[int],
        q_controls: list[int] | None = None,
        q_args: str | Iterable[str] | None = None,
    ):
        """
        Returns QASM string for gate definition or gate application given
        name, registers, arguments.
        """

        if not q_controls:
            q_controls = []
        q_regs = q_controls + q_targets

        if type(q_targets[0]) is int:
            q_regs = ",".join([f"q[{reg}]" for reg in q_regs])
        else:
            q_regs = ",".join(q_regs)

        if q_args:
            if isinstance(q_args, Iterable):
                q_args = ",".join([str(arg) for arg in q_args])
            return f"{q_name}({q_args}) {q_regs};"
        else:
            return f"{q_name} {q_regs};"

    def _qasm_defns(self, gate: Gate | Type[Gate]) -> None:
        """
        Define QASM gates for QuTiP gates that do not have QASM counterparts.

        Parameters
        ----------
        gate: :class:`~.operations.Gate`
            QuTiP gate which needs to be defined in QASM format.
        """

        if type(gate) is gates.CRY:
            gate_def = "gate cry(theta) a,b { cu3(theta,0,0) a,b; }"
        elif type(gate) is gates.CRX:
            gate_def = "gate crx(theta) a,b { cu3(theta,-pi/2,pi/2) a,b; }"
        elif gate == gates.SQRTX:
            gate_def = "gate sqrtx a {h a; u1(pi/2) a; h a; }"
        elif gate == gates.CZ:
            gate_def = "gate cz a,b { cu1(pi) a,b; }"
        elif gate == gates.CS:
            gate_def = "gate cs a,b { cu1(pi/2) a,b; }"
        elif gate == gates.CT:
            gate_def = "gate ct a,b { cu1(pi/4) a,b; }"
        elif gate == gates.SWAP:
            gate_def = "gate swap a,b { cx a,b; cx b,a; cx a,b; }"
        else:
            err_msg = f"No definition specified for {gate.name} gate"
            raise NotImplementedError(err_msg)

        self.output(f"// QuTiP definition for gate {gate.name}")
        self.output(gate_def)
        self.gate_name_map[gate.name] = gate.name.lower()

    def qasm_name(self, gate_name: str) -> str | None:
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

    def _qasm_output(self, qc: QubitCircuit) -> list[str]:
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
            raise NotImplementedError("QASM: Only OpenQASM 2.0 \
                                      is currently supported.")

        self.output('include "qelib1.inc";', 1)
        qc._to_qasm(self)

        return self.lines
