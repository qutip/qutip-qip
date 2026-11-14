"Decompose gates in CNOT + 1 qubit Rotation (RX, RY, RZ)"

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

PI = np.pi


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


@register_op(SQRTX)
def _gate_SQRTNOT(op, bloq):
    bloq.add_global_phase(PI / 4)
    bloq.add(RX(PI / 2), 0)


@register_op(H)
def _gate_H(op, bloq):
    bloq.add_global_phase(PI / 2)
    bloq.add(RY(PI / 2), 0)
    bloq.add(RX(PI), 0)


_gate_SNOT = _gate_H


@register_op(PHASE)
def _gate_PHASEGATE(op, bloq):
    bloq.add_global_phase(op.theta / 2)
    bloq.add(RZ(op.theta), 0)


@register_op(CZ)
def _gate_CZ(op, bloq):
    bloq.add_global_phase(PI)
    bloq.add(RY(PI / 2), 1)
    bloq.add(RX(PI), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RY(PI / 2), 1)
    bloq.add(RX(PI), 1)


_gate_CSIGN = _gate_CZ


@register_op(SWAP)
def _gate_SWAP(op, bloq):
    bloq.add(CX, (0, 1))
    bloq.add(CX, (1, 0))
    bloq.add(CX, (0, 1))


@register_op(ISWAP)
def _gate_ISWAP(op, bloq):
    bloq.add_global_phase(3 * PI / 2)
    bloq.add(SWAP, (0, 1))
    bloq.add(RZ(PI / 2), 1)
    bloq.add(RZ(PI / 2), 0)
    bloq.add(RY(PI / 2), 0)
    bloq.add(RX(PI), 0)
    bloq.add(CX, (1, 0))
    bloq.add(RY(PI / 2), 0)
    bloq.add(RX(PI), 0)


@register_op(FREDKIN)
def _gate_FREDKIN(op, bloq):
    bloq.add_global_phase(PI / 8)
    bloq.add(CX, (2, 1))
    bloq.add(RZ(PI), 2)
    bloq.add(RX(PI / 2), 2)
    bloq.add(RZ(-PI / 2), 2)
    bloq.add(RX(PI / 2), 2)
    bloq.add(RZ(PI), 2)
    bloq.add(CX, (1, 2))
    bloq.add(RZ(-PI / 4), 2)
    bloq.add(CX, (0, 2))
    bloq.add(RZ(PI / 4), 2)
    bloq.add(CX, (1, 2))
    bloq.add(RZ(PI / 4), 1)
    bloq.add(RZ(-PI / 4), 2)
    bloq.add(CX, (0, 2))
    bloq.add(CX, (0, 1))
    bloq.add(RZ(PI / 4), 0)
    bloq.add(RZ(-PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RZ(-3 * PI / 4), 2)
    bloq.add(RX(PI / 2), 2)
    bloq.add(RZ(-PI / 2), 2)
    bloq.add(RX(PI / 2), 2)
    bloq.add(RZ(PI), 2)
    bloq.add(CX, (2, 1))


@register_op(TOFFOLI)
def _gate_TOFFOLI(op, bloq):
    *ctrls, target = bloq.qreg

    bloq.add_global_phase(9 * PI / 8)
    bloq.add(RZ(PI / 4), ctrls[0])
    bloq.add(CX, ctrls)
    bloq.add(RZ(-PI / 4), ctrls[1])
    bloq.add(CX, ctrls)
    bloq.add(RY(PI / 2), target)
    bloq.add(RX(PI), target)
    bloq.add(RZ(PI / 4), ctrls[1])
    bloq.add(RZ(PI / 4), target)
    bloq.add(CX, (ctrls[0], target))
    bloq.add(RZ(-PI / 4), target)
    bloq.add(CX, (ctrls[1], target))
    bloq.add(RZ(PI / 4), target)
    bloq.add(CX, (ctrls[0], target))
    bloq.add(RZ(-PI / 4), target)
    bloq.add(CX, (ctrls[1], target))
    bloq.add(RY(PI / 2), target)
    bloq.add(RX(PI), target)
