"""
This is a temporary file.
Those decomposition functions should be moved to the
individual gate classes.
"""

import numpy as np

from qutip_qip.operations import register_op
from qutip_qip.operations.gates import (
    H,
    PHASE,
    RX,
    RY,
    RZ,
    CX,
    ISWAP,
    SQRTSWAP,
    SQRTISWAP,
    SQRTX,
    CZ,
    SWAP,
    FREDKIN,
    TOFFOLI,
)

__all__ = ["_resolve_to_universal", "_resolve_2q_basis"]


def _basis_CZ(qc_temp, temp_resolved):
    half_pi = PI / 2
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_gate(
                gate=RY(-half_pi),
                targets=targets,
            )
            qc_temp.add_gate(CZ, targets=targets, controls=controls)
            qc_temp.add_gate(
                gate=RY(half_pi),
                targets=targets,
            )
        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
            )


_basis_CSIGN = _basis_CZ


def _basis_ISWAP(qc_temp, temp_resolved):
    half_pi = PI / 2
    quarter_pi = PI / 4
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_global_phase(phase=quarter_pi)
            qc_temp.add_gate(ISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RZ(-half_pi),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RY(-half_pi),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RZ(half_pi),
                targets=controls,
            )
            qc_temp.add_gate(ISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RY(-half_pi),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RZ(half_pi),
                targets=targets,
            )

        elif gate == SWAP:
            qc_temp.add_global_phase(phase=quarter_pi)
            qc_temp.add_gate(ISWAP, targets=targets)
            qc_temp.add_gate(
                gate=RX(-half_pi),
                targets=targets[0],
            )
            qc_temp.add_gate(ISWAP, targets=targets)
            qc_temp.add_gate(
                gate=RX(-half_pi),
                targets=targets[1],
            )
            qc_temp.add_gate(ISWAP, targets=targets)
            qc_temp.add_gate(
                gate=RX(-half_pi),
                targets=targets[0],
            )

        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
            )


def _basis_SQRTSWAP(qc_temp, temp_resolved):
    half_pi = PI / 2
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_gate(
                gate=RY(half_pi),
                targets=targets,
            )
            qc_temp.add_gate(SQRTSWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RZ(PI),
                targets=controls,
            )
            qc_temp.add_gate(SQRTSWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RZ(-half_pi),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RY(-half_pi),
                targets=targets,
            )
            qc_temp.add_gate(
                gate=RZ(-half_pi),
                targets=controls,
            )
        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
            )


def _basis_SQRTISWAP(qc_temp, temp_resolved):
    half_pi = PI / 2
    for circ_instruction in temp_resolved.instructions:
        gate = circ_instruction.operation
        targets = circ_instruction.targets
        controls = circ_instruction.controls

        if gate == CX:
            qc_temp.add_gate(
                gate=RY(-half_pi),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RX(half_pi),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RX(-half_pi),
                targets=targets,
            )
            qc_temp.add_gate(SQRTISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RX(PI),
                targets=controls,
            )
            qc_temp.add_gate(SQRTISWAP, targets=[controls[0], targets[0]])
            qc_temp.add_gate(
                gate=RY(half_pi),
                targets=controls,
            )
            qc_temp.add_gate(
                gate=RZ(PI),
                targets=controls,
            )
            qc_temp.add_global_phase(phase=7 / 4 * PI)

        else:
            qc_temp.add_gate(
                gate,
                targets=targets,
                controls=controls,
                classical_controls=circ_instruction.cbits,
                classical_control_value=circ_instruction.cbits_ctrl_value,
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
