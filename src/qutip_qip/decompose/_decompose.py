import numpy as np

from qutip_qip.decompose.cnot_rotation import CNOT_ROTATION_BASIS
from qutip_qip.operations import register_op
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

PI = np.pi

CZ_BASIS = CNOT_ROTATION_BASIS.scope()
ISWAP_BASIS = CNOT_ROTATION_BASIS.scope()
SQRTSWAP_BASIS = CNOT_ROTATION_BASIS.scope()
SQRTISWAP_BASIS = CNOT_ROTATION_BASIS.scope()


@register_op(CX, repository=CZ_BASIS)
def _CX(op, bloq):
    ctrl, target = bloq.qreg

    bloq.add(RY(-PI / 2), target)
    bloq.add(CZ, (ctrl, target))
    bloq.add(RY(PI / 2), target)


@register_op(CX, repository=ISWAP_BASIS)
def _CX(op, bloq):
    ctrl, target = bloq.qreg

    bloq.add_global_phase(PI / 4)
    bloq.add(ISWAP, (ctrl, target))
    bloq.add(RZ(-PI / 2), target)
    bloq.add(RY(-PI / 2), ctrl)
    bloq.add(RZ(PI / 2), ctrl)
    bloq.add(ISWAP, (ctrl, target))
    bloq.add(RY(-PI / 2), target)
    bloq.add(RZ(PI / 2), target)


@register_op(SWAP, repository=ISWAP_BASIS)
def _SWAP(op, bloq):
    bloq.add_global_phase(PI / 4)
    bloq.add(ISWAP, (0, 1))
    bloq.add(RX(-PI / 2), 0)
    bloq.add(ISWAP, (0, 1))
    bloq.add(RX(-PI / 2), 1)
    bloq.add(ISWAP, (0, 1))
    bloq.add(RX(-PI / 2), 0)


@register_op(CX, repository=SQRTSWAP_BASIS)
def _CX(op, bloq):
    ctrl, target = bloq.qreg

    bloq.add(RY(PI / 2), target)
    bloq.add(SQRTSWAP, (ctrl, target))
    bloq.add(RZ(PI), ctrl)
    bloq.add(SQRTSWAP, (ctrl, target))
    bloq.add(RZ(-PI / 2), target)
    bloq.add(RY(-PI / 2), target)
    bloq.add(RZ(-PI / 2), ctrl)


@register_op(CX, repository=SQRTISWAP_BASIS)
def _CX(op, bloq):
    ctrl, target = bloq.qreg

    bloq.add_global_phase(7 / 4 * PI)
    bloq.add(RY(-PI / 2), ctrl)
    bloq.add(RX(PI / 2), ctrl)
    bloq.add(RX(-PI / 2), target)
    bloq.add(SQRTISWAP, (ctrl, target))
    bloq.add(RX(PI), ctrl)
    bloq.add(SQRTISWAP, (ctrl, target))
    bloq.add(RY(PI / 2), ctrl)
    bloq.add(RZ(PI), ctrl)
