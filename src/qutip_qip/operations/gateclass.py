# annotations import won't be needed after minimum version becomes 3.14 (PEP 749)
from __future__ import annotations

from qutip import Qobj
from qutip_qip.operations.op import Op, op_factory
from qutip_qip.typing import Real


class AngleParametricGate(Op):
    """Parametric gate whose parameters are all real angles."""

    def validate_params(self) -> None:
        for v in self.params:
            if isinstance(v, bool) or not isinstance(
                v, Real
            ):  # bool is an int in Python, but not an angle from our perspective
                raise TypeError(f"{self.name}: parameters must be real, got {v!r}.")


@op_factory
def get_unitary_gate(name: str, U: Qobj) -> type[Op]:
    """
    Gate Factory for Custom Gate that wraps an arbitrary unitary matrix U.
    """

    _name = name
    if not U.isoper or U.dims[0] != U.dims[1]:
        raise ValueError(
            "U must be a square operator with equal input and output dims."
        )

    if not U.isunitary:
        raise ValueError("U must be unitary.")

    class _CustomGate(Op):
        name = _name
        num_qreg = len(U.dims[0])
        qreg_dim = tuple(U.dims[0])
        self_inverse = U == U.dag()

        @staticmethod
        def compute_qobj(dtype=U.dtype):
            return U.to(dtype)

    _CustomGate.__name__ = _CustomGate.__qualname__ = _name
    return _CustomGate
