from qutip_qip.circuit import QubitCircuit
from qutip_qip.qasm.qasm import QasmProcessor
from qutip_qip.qasm.tokenize_qasm import tokenize_qasm


def read_qasm(
    qasm_input: str,
    mode: str = "default",
    version: str = "2.0",
    strmode: bool = False,
) -> QubitCircuit:
    """
    Read OpenQASM intermediate representation
    (https://github.com/Qiskit/openqasm) and return
    a :class:`.QubitCircuit` and state inputs as specified in the
    QASM file.

    Parameters
    ----------
    qasm_input : str
        File location or String Input for QASM file to be imported. In case of
        string input, the parameter strmode must be True.
    mode : str
        Parsing mode for the qasm file.
        - "default": For predefined gates in qutip-qip, use the predefined
        version, otherwise use the custom gate defined in qelib1.inc.
        The predefined gate can usually be further processed (e.g. decomposed)
        within qutip-qip.
        - "predefined_only": Use only the predefined gates in qutip-qip.
        - "external_only": Use only the gate defined in qelib1.inc, except for CX and QASMU gate.
    version : str
        QASM version of the QASM file. Only version 2.0 is currently supported.
    strmode : bool
        if specified as True, indicates that qasm_input is in string format
        rather than from file.

    Returns
    -------
    qc : :class:`.QubitCircuit`
        Returns a :class:`.QubitCircuit` object specified in the QASM file.
    """

    if strmode:
        qasm_lines = qasm_input.splitlines()
    else:
        f = open(qasm_input, "r")
        qasm_lines = f.read().splitlines()
        f.close()

    # split input into lines and ignore comments
    qasm_lines = [line.strip() for line in qasm_lines]
    qasm_lines = list(filter(lambda x: x[:2] != "//" and x != "", qasm_lines))
    # QASMBench Benchmark Suite has lines that have comments after instructions.
    # Not sure if QASM standard allows this.
    for i in range(len(qasm_lines)):
        qasm_line = qasm_lines[i]
        loc_comment = qasm_line.find("//")
        if loc_comment >= 0:
            qasm_line = qasm_line[0:loc_comment]
        qasm_lines[i] = qasm_line

    if version != "2.0":
        raise NotImplementedError(
            "QASM: Only OpenQASM 2.0 \
                                  is currently supported."
        )

    if qasm_lines.pop(0) != "OPENQASM 2.0;":
        raise SyntaxError("QASM: File does not contain QASM 2.0 header")

    qasm_obj = QasmProcessor(qasm_lines, mode=mode, version=version)
    qasm_obj.commands = tokenize_qasm(qasm_obj.commands)

    qasm_obj._process_includes()

    qasm_obj._initialize_pass()
    qc = QubitCircuit(qasm_obj.num_qubits, num_cbits=qasm_obj.num_cbits)

    qasm_obj._final_pass(qc)

    return qc