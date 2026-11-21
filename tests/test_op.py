import pytest
import qutip

import qutip_qip.operations.gates as gates
from qutip_qip.algorithms import QFT
from qutip_qip.operations import Controlled, Inverse, Modifier, Power, Unitary

MODIFIERS = {
    "Inverse": Inverse,
    "Controlled": Controlled,
    "Power": lambda op: Power(op, 2),
}


class U1(Unitary):
    num_qreg = 1

    @staticmethod
    def compute_qobj(dtype: str = "dense"):
        return qutip.rand_unitary(dimensions=[2]).to(dtype)


@pytest.mark.parametrize("modify", MODIFIERS.values(), ids=MODIFIERS.keys())
def test_modifier_target(modify):
    class NoMatrix(Unitary):  # parametric, with no closed-form inverse
        theta: float
        num_qreg = 1

    assert isinstance(modify(U1), Modifier)
    assert modify(U1).target is U1
    assert modify(NoMatrix(0.5)).target == NoMatrix(0.5)
    assert modify(U1).has_matrix and not modify(NoMatrix(0.5)).has_matrix


def test_target_of_a_family_is_a_family():
    assert gates.CRX.target is gates.RX
    assert gates.CRX(0.5).target == gates.RX(0.5)
    assert Inverse(QFT).target is QFT
    assert Inverse(QFT)(3).target == QFT(3)
