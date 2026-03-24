"""
This is a temporary file.
Those decomposition functions should be moved to the
individual gate classes.
"""

import numpy as np
from qutip_qip.operations.gates import (
    RX,
    RY,
    RZ,
    CX,
    ISWAP,
    SQRTSWAP,
    SQRTISWAP,
    CZ,
    SWAP,
)

__all__ = ["_resolve_to_universal", "_resolve_2q_basis"]


def _gate_IGNORED(circ_instruction, temp_resolved):
    gate = circ_instruction.operation
    targets = circ_instruction.targets
    controls = circ_instruction.controls

    temp_resolved.add_gate(
        gate,
        targets=targets,
        controls=controls,
        classical_controls=circ_instruction.cbits,
        classical_control_value=circ_instruction.cbits_ctrl_value,
        style=circ_instruction.style,
    )


def _controlled_gate_IGNORED(circ_instruction, temp_resolved):
    temp_resolved.add_gate(
        gate=circ_instruction.operation,
        targets=circ_instruction.targets,
        controls=circ_instruction.controls,
    )


_gate_RX = _gate_RY = _gate_RZ = _gate_IGNORED
_gate_basis_2q = _gate_IDLE = _gate_IGNORED
_gate_CNOT = _gate_CX = _controlled_gate_IGNORED


def _gate_NOTIMPLEMENTED(instruction, temp_resolved):
    raise NotImplementedError("Cannot be resolved in this basis")


_gate_BERKELEY = _gate_NOTIMPLEMENTED
_gate_SWAPalpha = _gate_NOTIMPLEMENTED
_gate_SQRTSWAP = _gate_NOTIMPLEMENTED
_gate_SQRTISWAP = _gate_NOTIMPLEMENTED


def _gate_SQRTNOT(circ_instruction, temp_resolved):
    targets = circ_instruction.targets

    temp_resolved.add_global_phase(phase=np.pi / 4)
    temp_resolved.add_gate(
        RX(np.pi / 2, arg_label=r"\pi/2"),
        targets=targets,
    )


def _gate_H(circ_instruction, temp_resolved):
    half_pi = np.pi / 2
    targets = circ_instruction.targets

    temp_resolved.add_global_phase(phase=half_pi)
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets,
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets,
    )


_gate_SNOT = _gate_H


def _gate_PHASEGATE(circ_instruction, temp_resolved):
    gate = circ_instruction.operation
    targets = circ_instruction.targets

    temp_resolved.add_global_phase(phase=gate.arg_value[0] / 2)
    temp_resolved.add_gate(
        gate=RZ(gate.arg_value[0], arg_label=gate.arg_label),
        targets=targets,
    )


def _gate_CZ(circ_instruction, temp_resolved):
    half_pi = np.pi / 2
    targets = circ_instruction.targets
    controls = circ_instruction.controls

    temp_resolved.add_global_phase(phase=np.pi)
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets,
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets,
    )
    temp_resolved.add_gate(CX, targets=targets, controls=controls)
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets,
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets,
    )


_gate_CSIGN = _gate_CZ


def _gate_SWAP(circ_instruction, temp_resolved):
    targets = circ_instruction.targets

    temp_resolved.add_gate(CX, targets=targets[0], controls=targets[1])
    temp_resolved.add_gate(CX, targets=targets[1], controls=targets[0])
    temp_resolved.add_gate(CX, targets=targets[0], controls=targets[1])


def _gate_ISWAP(circ_instruction, temp_resolved):
    half_pi = np.pi / 2
    targets = circ_instruction.targets

    temp_resolved.add_global_phase(phase=3 * half_pi)
    temp_resolved.add_gate(CX, targets=targets[0], controls=targets[1])
    temp_resolved.add_gate(CX, targets=targets[1], controls=targets[0])
    temp_resolved.add_gate(CX, targets=targets[0], controls=targets[1])
    temp_resolved.add_gate(
        RZ(half_pi, arg_label=r"\pi/2"),
        targets=targets[0],
    )
    temp_resolved.add_gate(
        RZ(half_pi, arg_label=r"\pi/2"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets[0],
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets[0],
    )
    temp_resolved.add_gate(CX, targets=targets[0], controls=targets[1])
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets[0],
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets[0],
    )


def _gate_FREDKIN(circ_instruction, temp_resolved):
    pi = np.pi
    targets = circ_instruction.targets
    controls = circ_instruction.controls

    temp_resolved.add_gate(CX, controls=targets[1], targets=targets[0])
    temp_resolved.add_gate(
        RZ(pi, arg_label=r"\pi"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RX(pi / 2, arg_label=r"\pi/2"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RZ(-pi / 2, arg_label=r"-\pi/2"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RX(pi / 2, arg_label=r"\pi/2"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RZ(pi, arg_label=r"\pi"),
        targets=targets[1],
    )
    temp_resolved.add_gate(CX, controls=targets[0], targets=targets[1])
    temp_resolved.add_gate(
        RZ(-pi / 4, arg_label=r"-\pi/4"),
        targets=targets[1],
    )
    temp_resolved.add_gate(CX, controls=controls, targets=targets[1])
    temp_resolved.add_gate(
        RZ(pi / 4, arg_label=r"\pi/4"),
        targets=targets[1],
    )
    temp_resolved.add_gate(CX, controls=targets[0], targets=targets[1])
    temp_resolved.add_gate(
        RZ(pi / 4, arg_label=r"\pi/4"),
        targets=targets[0],
    )
    temp_resolved.add_gate(
        RZ(-pi / 4, arg_label=r"-\pi/4"),
        targets=targets[1],
    )
    temp_resolved.add_gate(CX, controls=controls, targets=targets[1])
    temp_resolved.add_gate(CX, controls=controls, targets=targets[0])
    temp_resolved.add_gate(
        RZ(pi / 4, arg_label=r"\pi/4"),
        targets=controls,
    )
    temp_resolved.add_gate(
        RZ(-pi / 4, arg_label=r"-\pi/4"),
        targets=targets[0],
    )
    temp_resolved.add_gate(CX, controls=controls, targets=targets[0])
    temp_resolved.add_gate(
        gate=RZ(-3 * pi / 4, arg_label=r"-3\pi/4"),
        targets=targets[1],
    )
    temp_resolved.add_gate(RX(pi / 2, arg_label=r"\pi/2"), targets=targets[1])
    temp_resolved.add_gate(
        RZ(-pi / 2, arg_label=r"-\pi/2"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RX(pi / 2, arg_label=r"\pi/2"),
        targets=targets[1],
    )
    temp_resolved.add_gate(
        RZ(pi, arg_label=r"\pi"),
        targets=targets[1],
    )
    temp_resolved.add_gate(CX, controls=targets[1], targets=targets[0])
    temp_resolved.add_global_phase(phase=pi / 8)


# TODO add a test for this
def _gate_TOFFOLI(circ_instruction, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4

    targets = circ_instruction.targets
    controls = circ_instruction.controls

    temp_resolved.add_global_phase(phase=np.pi / 8)
    temp_resolved.add_gate(
        RZ(half_pi, arg_label=r"\pi/2"),
        targets=controls[1],
    )
    temp_resolved.add_gate(
        RZ(quarter_pi, arg_label=r"\pi/4"),
        targets=controls[0],
    )
    temp_resolved.add_gate(CX, targets=controls[1], controls=controls[0])
    temp_resolved.add_gate(
        RZ(-quarter_pi, arg_label=r"-\pi/4"),
        targets=controls[1],
    )
    temp_resolved.add_gate(CX, targets=controls[1], controls=controls[0])
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets,
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets,
    )
    temp_resolved.add_gate(
        RZ(quarter_pi, arg_label=r"\pi/4"),
        targets=controls[1],
    )
    temp_resolved.add_gate(
        RZ(quarter_pi, arg_label=r"\pi/4"),
        targets=targets,
    )
    temp_resolved.add_gate(CX, targets=targets, controls=controls[0])
    temp_resolved.add_gate(
        RZ(-quarter_pi, arg_label=r"-\pi/4"),
        targets=targets,
    )
    temp_resolved.add_gate(CX, targets=targets, controls=controls[1])
    temp_resolved.add_gate(
        RZ(quarter_pi, arg_label=r"\pi/4"),
        targets=targets,
    )
    temp_resolved.add_gate(CX, targets=targets, controls=controls[0])
    temp_resolved.add_gate(
        RZ(-quarter_pi, arg_label=r"-\pi/4"),
        targets=targets,
    )
    temp_resolved.add_gate(CX, targets=targets, controls=controls[1])
    temp_resolved.add_gate(
        RY(half_pi, arg_label=r"\pi/2"),
        targets=targets,
    )
    temp_resolved.add_gate(
        RX(np.pi, arg_label=r"\pi"),
        targets=targets,
    )
    temp_resolved.add_global_phase(phase=np.pi)


def _basis_CZ(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_gate(
                gate=RY(-half_pi, arg_label=r"-\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(CZ, targets=targets, controls=controls)
            qc_temp.add_gate(
                gate=RY(half_pi, arg_label=r"\pi/2"),
                targets=targets,
            )
        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
                style=circ_instruction.style,
            )


_basis_CSIGN = _basis_CZ


def _basis_ISWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    quarter_pi = np.pi / 4
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_global_phase(phase=quarter_pi)
            qc_temp.add_gate(ISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RZ(-half_pi, arg_label=r"-\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RY(-half_pi, arg_label=r"-\pi/2"),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RZ(half_pi, arg_label=r"\pi/2"),
                targets=controls,
            )
            qc_temp.add_gate(ISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RY(-half_pi, arg_label=r"-\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RZ(half_pi, arg_label=r"\pi/2"),
                targets=targets,
            )

        elif gate == SWAP:
            qc_temp.add_global_phase(phase=quarter_pi)
            qc_temp.add_gate(ISWAP, targets=targets)
            qc_temp.add_gate(
                gate=RX(-half_pi, arg_label=r"-\pi/2"),
                targets=targets[0],
            )
            qc_temp.add_gate(ISWAP, targets=targets)
            qc_temp.add_gate(
                gate=RX(-half_pi, arg_label=r"-\pi/2"),
                targets=targets[1],
            )
            qc_temp.add_gate(ISWAP, targets=targets)
            qc_temp.add_gate(
                gate=RX(-half_pi, arg_label=r"-\pi/2"),
                targets=targets[0],
            )

        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
                style=circ_instruction.style,
            )


def _basis_SQRTSWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_gate(
                gate=RY(half_pi, arg_label=r"\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(SQRTSWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RZ(np.pi, arg_label=r"\pi"),
                targets=controls,
            )
            qc_temp.add_gate(SQRTSWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RZ(-half_pi, arg_label=r"-\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RY(-half_pi, arg_label=r"-\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RZ(-half_pi, arg_label=r"-\pi/2"),
                targets=controls,
            )
        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
                style=circ_instruction.style,
            )


def _basis_SQRTISWAP(qc_temp, temp_resolved):
    half_pi = np.pi / 2
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_gate(
                gate=RY(-half_pi, arg_label=r"-\pi/2"),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RX(half_pi, arg_label=r"\pi/2"),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RX(-half_pi, arg_label=r"-\pi/2"),
                targets=targets,
            )
            qc_temp.add_gate(SQRTISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RX(np.pi, arg_label=r"\pi"),
                targets=controls,
            )
            qc_temp.add_gate(SQRTISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RY(half_pi, arg_label=r"\pi/2"),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RZ(np.pi, arg_label=r"\pi"),
                targets=controls,
            )
            qc_temp.add_global_phase(phase=7 / 4 * np.pi)

        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
                style=circ_instruction.style,
            )


def _resolve_2q_basis(basis, qc_temp, temp_resolved):
    """Dispatch method"""
    method = globals()["_basis_" + str(basis)]
    method(qc_temp, temp_resolved)


def _resolve_to_universal(instruction, temp_resolved, basis_1q, basis_2q):
    """A dispatch method"""
    gate_name = instruction.operation.name

    if gate_name in basis_2q:
        method = _gate_basis_2q
    else:
        if gate_name == "SWAP" and "ISWAP" in basis_2q:
            method = _gate_IGNORED
        else:
            method = globals()["_gate_" + str(gate_name)]
    method(instruction, temp_resolved)
