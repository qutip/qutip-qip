import numpy as np
from qutip_qip.operations import Unitary, Controlled, register_op
from qutip_qip.operations.gates import H, X, Z, CZ


class PhaseOracle(Unitary):
    n: int
    marked: tuple[int, ...]
    self_inverse = True

    @property
    def num_qreg(self) -> int:
        return self.n


@register_op(PhaseOracle)
def _phase_oracle(op, bloq):
    if op.n == 1:
        mcz = Z
    else:
        mcz = Controlled(Z, op.n - 1)

    for m in op.marked:
        zeros = [q for q in range(op.n) if not (m >> (op.n - 1 - q)) & 1]
        for q in zeros:
            bloq.add(X, q)
        bloq.add(mcz, tuple(range(op.n)))
        for q in zeros:
            bloq.add(X, q)


class Diffusion(Unitary):
    n: int
    self_inverse = True

    @property
    def num_qreg(self) -> int:
        return self.n


@register_op(Diffusion)
def _diffusion(op, bloq):
    qreg = tuple()
    for q in range(op.n):
        bloq.add(H, q)

    bloq.add(PhaseOracle(op.n, (0,)), qreg)
    for q in qreg:
        bloq.add(H, q)

    # The H-X-MCZ-X-H sequence is minus the textbook diffusion operator.
    bloq.add_global_phase(np.pi)


class Grover(Unitary):
    oracle: Unitary
    num_iterations: int

    @property
    def num_qreg(self) -> int:
        return self.oracle.num_qreg


@register_op(Grover)
def _grover(op, bloq):
    qreg = tuple(range(op.num_qreg))
    for q in qreg:
        bloq.add(H, q)

    for _ in range(op.num_iterations):
        bloq.add(op.oracle, qreg)
        bloq.add(Diffusion(op.num_qreg), qreg)
