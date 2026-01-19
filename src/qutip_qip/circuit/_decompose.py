"""
This is a temporary file.
Those decomposition functions should be moved to the
individual gate classes.
"""

import numpy as np
from qutip_qip.operations import (
    GLOBALPHASE,
    RX,
    RY,
    RZ,
    CSIGN,
    CNOT,
)


__all__ = ["_resolve_to_universal", "_resolve_2q_basis"]


def _gate_IGNORED(gate, temp_resolved):
    temp_resolved.append(gate)


_gate_RY = _gate_RZ = _gate_basis_2q = _gate_IDLE = _gate_IGNORED
_gate_CNOT = _gate_RX = _gate_IGNORED


def _gate_SQRTNOT(gate, temp_resolved):
    temp_resolved.extend(
        [
            GLOBALPHASE(arg_value=np.pi / 4, arg_label=r"\pi/4"),
            RX(target=gate.targets, arg_value=np.pi / 2, arg_label=r"\pi/2"),
        ]
    )


def _gate_SNOT(gate, temp_resolved):
    half_pi = np.pi / 2
    temp_resolved.extend(
        [
            GLOBALPHASE(arg_value=half_pi, arg_label=r"\pi/2"),
            RY(targets=gate.targets, arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets, arg_value=np.pi, arg_label=r"\pi"),
        ]
    )


_gate_H = _gate_SNOT


def _gate_PHASEGATE(gate, temp_resolved):
    temp_resolved.append(
        GLOBALPHASE(arg_value=gate.arg_value / 2, arg_label=gate.arg_label)
    )
    temp_resolved.append(
        RZ(
            targets=gate.targets,
            arg_value=gate.arg_value,
            arg_label=gate.arg_label,
        )
    )


def _gate_NOTIMPLEMENTED(gate, temp_resolved):
    raise NotImplementedError("Cannot be resolved in this basis")


_gate_BERKELEY = _gate_NOTIMPLEMENTED
_gate_SWAPalpha = _gate_NOTIMPLEMENTED
_gate_SQRTSWAP = _gate_NOTIMPLEMENTED
_gate_SQRTISWAP = _gate_NOTIMPLEMENTED


def _gate_CSIGN(gate, temp_resolved):
    half_pi = np.pi / 2
    temp_resolved.extend(
        [
            RY(targets=gate.targets, arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets, arg_value=np.pi, arg_label=r"\pi"),
            CNOT(targets=gate.targets, controls=gate.controls),
            RY(targets=gate.targets, arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets, arg_value=np.pi, arg_label=r"\pi"),
            GLOBALPHASE(arg_value=np.pi, arg_label=r"\pi"),
        ]
    )


def _gate_SWAP(gate, temp_resolved):
    temp_resolved.extend(
        [
            CNOT(targets=gate.targets[0], controls=gate.targets[1]),
            CNOT(targets=gate.targets[1], controls=gate.targets[0]),
            CNOT(targets=gate.targets[0], controls=gate.targets[1]),
        ]
    )


# FIXME the below method needs to be removed
def _gate_GLOBALPHASE(gate, temp_resolved):
    return temp_resolved


def _gate_ISWAP(gate, temp_resolved):
    half_pi = np.pi / 2
    temp_resolved.extend(
        [
            CNOT(targets=gate.targets[0], controls=gate.targets[1]),
            CNOT(targets=gate.targets[1], controls=gate.targets[0]),
            CNOT(targets=gate.targets[0], controls=gate.targets[1]),
            RZ(targets=gate.targets[0], arg_value=half_pi, arg_label=r"\pi/2"),
            RZ(targets=gate.targets[1], arg_value=half_pi, arg_label=r"\pi/2"),
            RY(targets=gate.targets[0], arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets[0], arg_value=np.pi, arg_label=r"\pi"),
            CNOT(targets=gate.targets[0], controls=gate.targets[1]),
            RY(targets=gate.targets[0], arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets[0], arg_value=np.pi, arg_label=r"\pi"),
            GLOBALPHASE(arg_value=3 * half_pi, arg_label=r"\pi/2"),
        ]
    )


def _gate_FREDKIN(gate, temp_resolved):
    pi = np.pi
    temp_resolved += [
        CNOT(controls=gate.targets[1], targets=gate.targets[0]),
        RZ(targets=gate.targets[1], arg_value=pi, arg_label=r"\pi"),
        RX(targets=gate.targets[1], arg_value=pi / 2, arg_label=r"\pi/2"),
        RZ(targets=gate.targets[1], arg_value=-pi / 2, arg_label=r"-\pi/2"),
        RX(targets=gate.targets[1], arg_value=pi / 2, arg_label=r"\pi/2"),
        RZ(targets=gate.targets[1], arg_value=pi, arg_label=r"\pi"),
        CNOT(controls=gate.targets[0], targets=gate.targets[1]),
        RZ(targets=gate.targets[1], arg_value=-pi / 4, arg_label=r"-\pi/4"),
        CNOT(controls=gate.controls, targets=gate.targets[1]),
        RZ(targets=gate.targets[1], arg_value=pi / 4, arg_label=r"\pi/4"),
        CNOT(controls=gate.targets[0], targets=gate.targets[1]),
        RZ(targets=gate.targets[0], arg_value=pi / 4, arg_label=r"\pi/4"),
        RZ(targets=gate.targets[1], arg_value=-pi / 4, arg_label=r"-\pi/4"),
        CNOT(controls=gate.controls, targets=gate.targets[1]),
        CNOT(controls=gate.controls, targets=gate.targets[0]),
        RZ(targets=gate.controls, arg_value=pi / 4, arg_label=r"\pi/4"),
        RZ(targets=gate.targets[0], arg_value=-pi / 4, arg_label=r"-\pi/4"),
        CNOT(controls=gate.controls, targets=gate.targets[0]),
        RZ(
            targets=gate.targets[1],
            arg_value=-3 * pi / 4,
            arg_label=r"-3\pi/4",
        ),
        RX(targets=gate.targets[1], arg_value=pi / 2, arg_label=r"\pi/2"),
        RZ(targets=gate.targets[1], arg_value=-pi / 2, arg_label=r"-\pi/2"),
        RX(targets=gate.targets[1], arg_value=pi / 2, arg_label=r"\pi/2"),
        RZ(targets=gate.targets[1], arg_value=pi, arg_label=r"\pi"),
        CNOT(controls=gate.targets[1], targets=gate.targets[0]),
        GLOBALPHASE(arg_value=pi / 8, arg_label=r"\pi/8"),
    ]


def _gate_TOFFOLI(gate, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4
    temp_resolved.append(
        [
            GLOBALPHASE(arg_value=np.pi / 8, arg_label=r"\pi/8"),
            RZ(
                targets=gate.controls[1], arg_value=half_pi, arg_label=r"\pi/2"
            ),
            RZ(
                targets=gate.controls[0],
                arg_value=quarter_pi,
                arg_label=r"\pi/4",
            ),
            CNOT(targets=gate.controls[1], controls=gate.controls[0]),
            RZ(
                targets=gate.controls[1],
                arg_value=-quarter_pi,
                arg_label=r"-\pi/4",
            ),
            CNOT(targets=gate.controls[1], controls=gate.controls[0]),
            RY(targets=gate.targets, arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets, arg_value=np.pi, arg_label=r"\pi"),
            RZ(
                targets=gate.controls[1],
                arg_value=quarter_pi,
                arg_label=r"\pi/4",
            ),
            RZ(targets=gate.targets, arg_value=quarter_pi, arg_label=r"\pi/4"),
            CNOT(targets=gate.targets, controls=gate.controls[0]),
            RZ(
                targets=gate.targets,
                arg_value=-quarter_pi,
                arg_label=r"-\pi/4",
            ),
            CNOT(targets=gate.targets, controls=gate.controls[1]),
            RZ(targets=gate.targets, arg_value=quarter_pi, arg_label=r"\pi/4"),
            CNOT(targets=gate.targets, controls=gate.controls[0]),
            RZ(
                targets=gate.targets,
                arg_value=-quarter_pi,
                arg_label=r"-\pi/4",
            ),
            CNOT(targets=gate.targets, controls=gate.controls[1]),
            GLOBALPHASE(arg_value=np.pi, arg_label=r"\pi"),
            RY(targets=gate.targets, arg_value=half_pi, arg_label=r"\pi/2"),
            RX(targets=gate.targets, arg_value=np.pi, arg_label=r"\pi"),
        ]
    )


def _basis_CSIGN(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for gate in temp_resolved:
        if gate.name == "CNOT":
            qc_temp.add_gate(
                "RY",
                targets=gate.targets,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "CSIGN",
                targets=gate.targets,
                controls=gate.controls
            )
            qc_temp.add_gate(
                "RY",
                targets=gate.targets,
                arg_value=half_pi,
                arg_label=r"\pi/2",
            )
        else:
            qc_temp.gates.append(gate)


def _basis_ISWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4
    for gate in temp_resolved:

        if gate.name == "CNOT":
            qc_temp.add_gate(
                "GLOBALPHASE",
                arg_value=quarter_pi,
                arg_label=r"\pi/4",
            )
            qc_temp.add_gate("ISWAP", targets=[gate.controls[0], gate.targets[0]])
            qc_temp.add_gate(
                "RZ",
                targets=gate.targets,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "RY",
                targets=gate.controls,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "RZ",
                targets=gate.controls,
                arg_value=half_pi,
                arg_label=r"\pi/2",
            )
            qc_temp.add_gate("ISWAP", targets=[gate.controls[0], gate.targets[0]])
            qc_temp.add_gate(
                "RY",
                targets=gate.targets,
                arg_value=-half_pi,
                arg_label=r"-\pi/2"
            )
            qc_temp.add_gate(
                "RZ",
                targets=gate.targets,
                arg_value=half_pi,
                arg_label=r"\pi/2",
            )

        elif gate.name == "SWAP":
            qc_temp.add_gate(
                "GLOBALPHASE",
                arg_value=quarter_pi,
                arg_label=r"\pi/4",
            )
            qc_temp.add_gate("ISWAP", targets=gate.targets)
            qc_temp.add_gate(
                "RX",
                targets=gate.targets[0],
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate("ISWAP", targets=gate.targets)
            qc_temp.add_gate(
                "RX",
                targets=gate.targets[1],
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate("ISWAP", targets=gate.targets)
            qc_temp.add_gate(
                "RX",
                targets=gate.targets[0],
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            
        else:
            qc_temp.gates.append(gate)


def _basis_SQRTSWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for gate in temp_resolved:

        if gate.name == "CNOT":
            qc_temp.add_gate(
                "RY",
                targets=gate.targets,
                arg_value=half_pi,
                arg_label=r"\pi/2",
            )
            qc_temp.add_gate(
                "SQRTSWAP", targets=[gate.controls[0], gate.targets[0]]
            )
            qc_temp.add_gate(
                "RZ",
                targets=gate.controls,
                arg_value=np.pi,
                arg_label=r"\pi",
            )
            qc_temp.add_gate(
                "SQRTSWAP", targets=[gate.controls[0], gate.targets[0]]
            )
            qc_temp.add_gate(
                "RZ",
                targets=gate.targets,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "RY",
                targets=gate.targets,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "RZ",
                targets=gate.controls,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
        else:
            qc_temp.gates.append(gate)


def _basis_SQRTISWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for gate in temp_resolved:

        if gate.name == "CNOT":
            qc_temp.add_gate(
                "RY",
                targets=gate.controls,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "RX",
                targets=gate.controls,
                arg_value=half_pi,
                arg_label=r"\pi/2",
            )
            qc_temp.add_gate(
                "RX",
                targets=gate.targets,
                arg_value=-half_pi,
                arg_label=r"-\pi/2",
            )
            qc_temp.add_gate(
                "SQRTISWAP", targets=[gate.controls[0], gate.targets[0]]
            )
            qc_temp.add_gate(
                "RX",
                targets=gate.controls,
                arg_value=np.pi,
                arg_label=r"\pi",
            )
            qc_temp.add_gate(
                "SQRTISWAP", targets=[gate.controls[0], gate.targets[0]]
            )
            qc_temp.add_gate(
                "RY",
                targets=gate.controls,
                arg_value=half_pi,
                arg_label=r"\pi/2",
            )
            qc_temp.add_gate(
                "RZ",
                targets=gate.controls,
                arg_value=np.pi,
                arg_label=r"\pi",
            )
            qc_temp.add_gate(
                "GLOBALPHASE",
                arg_value=7/4 * np.pi,
                arg_label=r"7\pi/4",
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
