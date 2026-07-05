from qutip_qip.operations import Op


class Label(Op):
    """A static marker in the instruction list."""

    def __init__(self, name):
        self.name = name


class Conditional(Op):
    """Classical conditional control flow statements"""

    def __init__(self, label):
        self.label = label


class Cbz(Conditional):
    "Conditional branch on zero"


class Cbnz(Conditional):
    "Conditional branch on non-zero i.e. 1."
