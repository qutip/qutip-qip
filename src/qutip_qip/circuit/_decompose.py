"""
This is a temporary file.
Those decomposition functions should be moved to the
individual gate classes.
"""

import numpy as np
from ..operations import Gate


__all__ = ["_resolve_to_universal", "_resolve_2q_basis"]


def _gate_IGNORED(gate, temp_resolved):
    temp_resolved.append(gate)


_gate_RY = _gate_RZ = _gate_basis_2q = _gate_IDLE = _gate_IGNORED
_gate_CNOT = _gate_RX = _gate_IGNORED


def _gate_SQRTNOT(gate, temp_resolved):
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=np.pi / 4,
            arg_label=r"\pi/4",
        )
    )
    temp_resolved.append(
        Gate(
            "RX",
            gate.targets,
            None,
            arg_value=np.pi / 2,
            arg_label=r"\pi/2",
        )
    )


def _gate_SNOT(gate, temp_resolved):
    half_pi = np.pi / 2
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate("RY", gate.targets, None, arg_value=half_pi, arg_label=r"\pi/2")
    )
    temp_resolved.append(
        Gate("RX", gate.targets, None, arg_value=np.pi, arg_label=r"\pi")
    )


_gate_H = _gate_SNOT


def _gate_PHASEGATE(gate, temp_resolved):
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=gate.arg_value / 2,
            arg_label=gate.arg_label,
        )
    )
    temp_resolved.append(
        Gate("RZ", gate.targets, None, gate.arg_value, gate.arg_label)
    )


def _gate_NOTIMPLEMENTED(gate, temp_resolved):
    raise NotImplementedError("Cannot be resolved in this basis")


_gate_BERKELEY = _gate_NOTIMPLEMENTED
_gate_SWAPalpha = _gate_NOTIMPLEMENTED
_gate_SQRTSWAP = _gate_NOTIMPLEMENTED
_gate_SQRTISWAP = _gate_NOTIMPLEMENTED


def _gate_CSIGN(gate, temp_resolved):
    half_pi = np.pi / 2
    temp_resolved.append(
        Gate("RY", gate.targets, None, arg_value=half_pi, arg_label=r"\pi/2")
    )
    temp_resolved.append(
        Gate("RX", gate.targets, None, arg_value=np.pi, arg_label=r"\pi")
    )
    temp_resolved.append(Gate("CNOT", gate.targets, gate.controls))
    temp_resolved.append(
        Gate("RY", gate.targets, None, arg_value=half_pi, arg_label=r"\pi/2")
    )
    temp_resolved.append(
        Gate("RX", gate.targets, None, arg_value=np.pi, arg_label=r"\pi")
    )
    temp_resolved.append(
        Gate("GLOBALPHASE", None, None, arg_value=np.pi, arg_label=r"\pi")
    )


def _gate_SWAP(gate, temp_resolved):
    temp_resolved.append(Gate("CNOT", gate.targets[0], gate.targets[1]))
    temp_resolved.append(Gate("CNOT", gate.targets[1], gate.targets[0]))
    temp_resolved.append(Gate("CNOT", gate.targets[0], gate.targets[1]))


def _gate_ISWAP(gate, temp_resolved):
    half_pi = np.pi / 2
    temp_resolved.append(Gate("CNOT", gate.targets[0], gate.targets[1]))
    temp_resolved.append(Gate("CNOT", gate.targets[1], gate.targets[0]))
    temp_resolved.append(Gate("CNOT", gate.targets[0], gate.targets[1]))
    temp_resolved.append(
        Gate(
            "RZ",
            gate.targets[0],
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate(
            "RZ",
            gate.targets[1],
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate(
            "RY",
            gate.targets[0],
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate("RX", gate.targets[0], None, arg_value=np.pi, arg_label=r"\pi")
    )
    temp_resolved.append(Gate("CNOT", gate.targets[0], gate.targets[1]))
    temp_resolved.append(
        Gate(
            "RY",
            gate.targets[0],
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate("RX", gate.targets[0], None, arg_value=np.pi, arg_label=r"\pi")
    )
    temp_resolved.append(
        Gate("GLOBALPHASE", None, None, arg_value=np.pi, arg_label=r"\pi")
    )
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )


def _gate_FREDKIN(gate, temp_resolved):
    pi = np.pi
    temp_resolved += [
        Gate("CNOT", controls=gate.targets[1], targets=gate.targets[0]),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi,
            arg_label=r"\pi",
        ),
        Gate(
            "RX",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi / 2,
            arg_label=r"\pi/2",
        ),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=-pi / 2,
            arg_label=r"-\pi/2",
        ),
        Gate(
            "RX",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi / 2,
            arg_label=r"\pi/2",
        ),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi,
            arg_label=r"\pi",
        ),
        Gate("CNOT", controls=gate.targets[0], targets=gate.targets[1]),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=-pi / 4,
            arg_label=r"-\pi/4",
        ),
        Gate("CNOT", controls=gate.controls, targets=gate.targets[1]),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi / 4,
            arg_label=r"\pi/4",
        ),
        Gate("CNOT", controls=gate.targets[0], targets=gate.targets[1]),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[0],
            arg_value=pi / 4,
            arg_label=r"\pi/4",
        ),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=-pi / 4,
            arg_label=r"-\pi/4",
        ),
        Gate("CNOT", controls=gate.controls, targets=gate.targets[1]),
        Gate("CNOT", controls=gate.controls, targets=gate.targets[0]),
        Gate(
            "RZ",
            controls=None,
            targets=gate.controls,
            arg_value=pi / 4,
            arg_label=r"\pi/4",
        ),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[0],
            arg_value=-pi / 4,
            arg_label=r"-\pi/4",
        ),
        Gate("CNOT", controls=gate.controls, targets=gate.targets[0]),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=-3 * pi / 4,
            arg_label=r"-3\pi/4",
        ),
        Gate(
            "RX",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi / 2,
            arg_label=r"\pi/2",
        ),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=-pi / 2,
            arg_label=r"-\pi/2",
        ),
        Gate(
            "RX",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi / 2,
            arg_label=r"\pi/2",
        ),
        Gate(
            "RZ",
            controls=None,
            targets=gate.targets[1],
            arg_value=pi,
            arg_label=r"\pi",
        ),
        Gate("CNOT", controls=gate.targets[1], targets=gate.targets[0]),
        Gate(
            "GLOBALPHASE",
            controls=None,
            targets=None,
            arg_value=pi / 8,
            arg_label=r"\pi/8",
        ),
    ]


def _gate_TOFFOLI(gate, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=np.pi / 8,
            arg_label=r"\pi/8",
        )
    )
    temp_resolved.append(
        Gate(
            "RZ",
            gate.controls[1],
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate(
            "RZ",
            gate.controls[0],
            None,
            arg_value=quarter_pi,
            arg_label=r"\pi/4",
        )
    )
    temp_resolved.append(Gate("CNOT", gate.controls[1], gate.controls[0]))
    temp_resolved.append(
        Gate(
            "RZ",
            gate.controls[1],
            None,
            arg_value=-quarter_pi,
            arg_label=r"-\pi/4",
        )
    )
    temp_resolved.append(Gate("CNOT", gate.controls[1], gate.controls[0]))
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate("RY", gate.targets, None, arg_value=half_pi, arg_label=r"\pi/2")
    )
    temp_resolved.append(
        Gate("RX", gate.targets, None, arg_value=np.pi, arg_label=r"\pi")
    )
    temp_resolved.append(
        Gate(
            "RZ",
            gate.controls[1],
            None,
            arg_value=-quarter_pi,
            arg_label=r"-\pi/4",
        )
    )
    temp_resolved.append(
        Gate(
            "RZ",
            gate.targets,
            None,
            arg_value=quarter_pi,
            arg_label=r"\pi/4",
        )
    )
    temp_resolved.append(Gate("CNOT", gate.targets, gate.controls[0]))
    temp_resolved.append(
        Gate(
            "RZ",
            gate.targets,
            None,
            arg_value=-quarter_pi,
            arg_label=r"-\pi/4",
        )
    )
    temp_resolved.append(Gate("CNOT", gate.targets, gate.controls[1]))
    temp_resolved.append(
        Gate(
            "RZ",
            gate.targets,
            None,
            arg_value=quarter_pi,
            arg_label=r"\pi/4",
        )
    )
    temp_resolved.append(Gate("CNOT", gate.targets, gate.controls[0]))
    temp_resolved.append(
        Gate(
            "RZ",
            gate.targets,
            None,
            arg_value=-quarter_pi,
            arg_label=r"-\pi/4",
        )
    )
    temp_resolved.append(Gate("CNOT", gate.targets, gate.controls[1]))
    temp_resolved.append(
        Gate(
            "GLOBALPHASE",
            None,
            None,
            arg_value=half_pi,
            arg_label=r"\pi/2",
        )
    )
    temp_resolved.append(
        Gate("RY", gate.targets, None, arg_value=half_pi, arg_label=r"\pi/2")
    )
    temp_resolved.append(
        Gate("RX", gate.targets, None, arg_value=np.pi, arg_label=r"\pi")
    )


def _gate_GLOBALPHASE(gate, temp_resolved):
    temp_resolved.append(
        Gate(
            gate.name,
            gate.targets,
            gate.controls,
            gate.arg_value,
            gate.arg_label,
        )
    )


def _basis_CSIGN(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for gate in temp_resolved:
        if gate.name == "CNOT":
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.targets,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(Gate("CSIGN", gate.targets, gate.controls))
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.targets,
                    None,
                    arg_value=half_pi,
                    arg_label=r"\pi/2",
                )
            )
        else:
            qc_temp.gates.append(gate)


def _basis_ISWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4
    for gate in temp_resolved:
        if gate.name == "CNOT":
            qc_temp.gates.append(
                Gate(
                    "GLOBALPHASE",
                    None,
                    None,
                    arg_value=quarter_pi,
                    arg_label=r"\pi/4",
                )
            )
            qc_temp.gates.append(
                Gate("ISWAP", [gate.controls[0], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.targets,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.controls,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.controls,
                    None,
                    arg_value=half_pi,
                    arg_label=r"\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate("ISWAP", [gate.controls[0], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.targets,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.targets,
                    None,
                    arg_value=half_pi,
                    arg_label=r"\pi/2",
                )
            )
        elif gate.name == "SWAP":
            qc_temp.gates.append(
                Gate(
                    "GLOBALPHASE",
                    None,
                    None,
                    arg_value=quarter_pi,
                    arg_label=r"\pi/4",
                )
            )
            qc_temp.gates.append(Gate("ISWAP", gate.targets, None))
            qc_temp.gates.append(
                Gate(
                    "RX",
                    gate.targets[0],
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(Gate("ISWAP", gate.targets, None))
            qc_temp.gates.append(
                Gate(
                    "RX",
                    gate.targets[1],
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate("ISWAP", [gate.targets[1], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RX",
                    gate.targets[0],
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
        else:
            qc_temp.gates.append(gate)


def _basis_SQRTSWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for gate in temp_resolved:
        if gate.name == "CNOT":
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.targets,
                    None,
                    arg_value=half_pi,
                    arg_label=r"\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate("SQRTSWAP", [gate.controls[0], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.controls,
                    None,
                    arg_value=np.pi,
                    arg_label=r"\pi",
                )
            )
            qc_temp.gates.append(
                Gate("SQRTSWAP", [gate.controls[0], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.targets,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.targets,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.controls,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
        else:
            qc_temp.gates.append(gate)


def _basis_SQRTISWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4
    for gate in temp_resolved:
        if gate.name == "CNOT":
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.controls,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RX",
                    gate.controls,
                    None,
                    arg_value=half_pi,
                    arg_label=r"\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RX",
                    gate.targets,
                    None,
                    arg_value=-half_pi,
                    arg_label=r"-\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate("SQRTISWAP", [gate.controls[0], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RX",
                    gate.controls,
                    None,
                    arg_value=np.pi,
                    arg_label=r"\pi",
                )
            )
            qc_temp.gates.append(
                Gate("SQRTISWAP", [gate.controls[0], gate.targets[0]], None)
            )
            qc_temp.gates.append(
                Gate(
                    "RY",
                    gate.controls,
                    None,
                    arg_value=half_pi,
                    arg_label=r"\pi/2",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "GLOBALPHASE",
                    None,
                    None,
                    arg_value=quarter_pi,
                    arg_label=r"\pi/4",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "RZ",
                    gate.controls,
                    None,
                    arg_value=np.pi,
                    arg_label=r"\pi",
                )
            )
            qc_temp.gates.append(
                Gate(
                    "GLOBALPHASE",
                    None,
                    None,
                    arg_value=3 * half_pi,
                    arg_label=r"3\pi/2",
                )
            )
        else:
            qc_temp.gates.append(gate)


def _resolve_2q_basis(basis, qc_temp, temp_resolved):
    """Dispatch method"""
    method = globals()["_basis_" + str(basis)]
    method(qc_temp, temp_resolved)


def _resolve_to_universal(gate, temp_resolved, basis_1q, basis_2q):
    """A dispatch method"""
    if gate.name in basis_2q:
        method = _gate_basis_2q
    else:
        if gate.name == "SWAP" and "ISWAP" in basis_2q:
            method = _gate_IGNORED
        else:
            method = globals()["_gate_" + str(gate.name)]
    method(gate, temp_resolved)
