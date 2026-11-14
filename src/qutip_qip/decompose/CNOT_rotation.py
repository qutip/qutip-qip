"Decompose gates in CNOT + 1 qubit Rotation (RX, RY, RZ)"

import numpy as np

from qutip_qip.operations import register_op
from qutip_qip.operations.gates import (
    X,
    Y,
    Z,
    S,
    Sdag,
    T,
    Tdag,
    SQRTXdag,
    R,
    QASMU,
    H,
    PHASE,
    RX,
    RY,
    RZ,
    CX,
    MS,
    RZX,
    ISWAP,
    ISWAPdag,
    SQRTSWAP,
    SQRTSWAPdag,
    SQRTISWAP,
    SQRTISWAPdag,
    SQRTX,
    CZ,
    SWAP,
    FREDKIN,
    TOFFOLI,
    BERKELEY,
    BERKELEYdag,
    SWAPALPHA,
)

PI = np.pi


# Single qubit
@register_op(X)
def _X(op, bloq):
    bloq.add_global_phase(PI / 2)
    bloq.add(RX(PI), 0)


@register_op(Y)
def _Y(op, bloq):
    bloq.add_global_phase(PI / 2)
    bloq.add(RY(PI), 0)


@register_op(Z)
def _Z(op, bloq):
    bloq.add_global_phase(PI / 2)
    bloq.add(RZ(PI), 0)


@register_op(H)
def _H(op, bloq):
    bloq.add_global_phase(PI / 2)
    bloq.add(RY(PI / 2), 0)
    bloq.add(RX(PI), 0)


@register_op(S)
def _S(op, bloq):
    bloq.add_global_phase(PI / 4)
    bloq.add(RZ(PI / 2), 0)


@register_op(Sdag)
def _Sdag(op, bloq):
    bloq.add_global_phase(-PI / 4)
    bloq.add(RZ(-PI / 2), 0)


@register_op(T)
def _T(op, bloq):
    bloq.add_global_phase(PI / 8)
    bloq.add(RZ(PI / 4), 0)


@register_op(Tdag)
def _Tdag(op, bloq):
    bloq.add_global_phase(-PI / 8)
    bloq.add(RZ(-PI / 4), 0)


@register_op(SQRTX)
def _SQRTNOT(op, bloq):
    bloq.add_global_phase(PI / 4)
    bloq.add(RX(PI / 2), 0)


@register_op(SQRTXdag)
def _SQRTXdag(op, bloq):
    bloq.add_global_phase(-PI / 4)
    bloq.add(RX(-PI / 2), 0)


@register_op(R)
def _R(op, bloq):
    bloq.add(RZ(-op.phi), 0)
    bloq.add(RX(op.theta), 0)
    bloq.add(RZ(op.phi), 0)


@register_op(QASMU)
def _QASMU(op, bloq):
    bloq.add(RZ(op.gamma), 0)
    bloq.add(RY(op.theta), 0)
    bloq.add(RZ(op.phi), 0)


@register_op(PHASE)
def _PHASE(op, bloq):
    bloq.add_global_phase(op.theta / 2)
    bloq.add(RZ(op.theta), 0)


# Two qubit-gates
@register_op(CZ)
def _CZ(op, bloq):
    bloq.add_global_phase(PI)
    bloq.add(RY(PI / 2), 1)
    bloq.add(RX(PI), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RY(PI / 2), 1)
    bloq.add(RX(PI), 1)


@register_op(SWAP)
def _SWAP(op, bloq):
    bloq.add(CX, (0, 1))
    bloq.add(CX, (1, 0))
    bloq.add(CX, (0, 1))


@register_op(ISWAP)
def _ISWAP(op, bloq):
    bloq.add(RX(PI / 2), 0)
    bloq.add(RX(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RZ(-PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RX(-PI / 2), 1)


@register_op(ISWAPdag)
def _ISWAP(op, bloq):
    bloq.add(RX(PI / 2), 0)
    bloq.add(RX(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(PI / 2), 0)
    bloq.add(RZ(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RX(-PI / 2), 1)


@register_op(MS)
def _MS(op, bloq):
    bloq.add(RZ(-op.phi), 0)
    bloq.add(RZ(-op.phi), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(op.theta), 0)
    bloq.add(CX, (0, 1))
    bloq.add(RZ(op.phi), 0)
    bloq.add(RZ(op.phi), 1)


@register_op(RZX)
def _RZX(op, bloq):
    bloq.add(RY(-PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RZ(op.theta), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RY(PI / 2), 1)


@register_op(SWAPALPHA)
def _SWAPALPHA(op, bloq):
    theta = PI / 2 * (1 + op.alpha)

    bloq.add_global_phase(theta / 2)
    bloq.add(RZ(-PI / 2), 1)
    bloq.add(CX, (1, 0))
    bloq.add(RZ(theta), 0)
    bloq.add(RY(-theta), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RY(theta), 1)
    bloq.add(CX, (1, 0))
    bloq.add(RZ(PI / 2), 0)


@register_op(SQRTSWAP)
def _SQRTSWAP(op, bloq):
    bloq.add_global_phase(3 * PI / 8)
    bloq.add(RZ(-PI / 2), 1)
    bloq.add(CX, (1, 0))
    bloq.add(RZ(3 * PI / 4), 0)
    bloq.add(RY(-3 * PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RY(3 * PI / 4), 1)
    bloq.add(CX, (1, 0))
    bloq.add(RZ(PI / 2), 0)


@register_op(SQRTSWAPdag)
def _SQRTSWAPdag(op, bloq):
    bloq.add_global_phase(PI / 8)
    bloq.add(RZ(-PI / 2), 1)
    bloq.add(CX, (1, 0))
    bloq.add(RZ(PI / 4), 0)
    bloq.add(RY(-PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RY(PI / 4), 1)
    bloq.add(CX, (1, 0))
    bloq.add(RZ(PI / 2), 0)


@register_op(SQRTISWAP)
def _SQRTISWAP(op, bloq):
    bloq.add(RX(PI / 2), 0)
    bloq.add(RX(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 4), 0)
    bloq.add(RZ(-PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RX(-PI / 2), 1)


@register_op(SQRTISWAPdag)
def _SQRTISWAPdag(op, bloq):
    bloq.add(RX(PI / 2), 0)
    bloq.add(RX(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(PI / 4), 0)
    bloq.add(RZ(PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RX(-PI / 2), 1)


@register_op(BERKELEY)
def _BERKELEY(op, bloq):
    bloq.add(RX(PI / 2), 0)
    bloq.add(RX(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RZ(-PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RX(-PI / 2), 1)


@register_op(BERKELEYdag)
def _BERKELEYdag(op, bloq):
    bloq.add(RX(PI / 2), 0)
    bloq.add(RX(PI / 2), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(PI / 2), 0)
    bloq.add(RZ(PI / 4), 1)
    bloq.add(CX, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(RX(-PI / 2), 1)


# Three qubit gates
@register_op(FREDKIN)
def _FREDKIN(op, bloq):
    ctrl, *target = bloq.qreg  # 1 control and 2 target

    bloq.add_global_phase(PI / 8)
    bloq.add(CX, (target[1], target[0]))
    bloq.add(RZ(PI), target[1])
    bloq.add(RX(PI / 2), target[1])
    bloq.add(RZ(-PI / 2), target[1])
    bloq.add(RX(PI / 2), target[1])
    bloq.add(RZ(PI), target[1])
    bloq.add(CX, target)
    bloq.add(RZ(-PI / 4), target[1])
    bloq.add(CX, (ctrl, target[1]))
    bloq.add(RZ(PI / 4), target[1])
    bloq.add(CX, target[1])
    bloq.add(RZ(PI / 4), target[0])
    bloq.add(RZ(-PI / 4), target[1])
    bloq.add(CX, (ctrl, target[1]))
    bloq.add(CX, (ctrl, target[1]))
    bloq.add(RZ(PI / 4), ctrl)
    bloq.add(RZ(-PI / 4), target[0])
    bloq.add(CX, (ctrl, target[0]))
    bloq.add(RZ(-3 * PI / 4), target[1])
    bloq.add(RX(PI / 2), target[1])
    bloq.add(RZ(-PI / 2), target[1])
    bloq.add(RX(PI / 2), target[1])
    bloq.add(RZ(PI), target[1])
    bloq.add(CX, (target[1], target[0]))


@register_op(TOFFOLI)
def _TOFFOLI(op, bloq):
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
