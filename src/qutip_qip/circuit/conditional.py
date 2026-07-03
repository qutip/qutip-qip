from qutip_qip.operations import Op


class Label(Op):
    """A static marker in the instruction list."""

    def __init__(self, name):
        self.name = name


class Cbz(Op):
    "Conditional branch on zero"

    def __init__(self, label):
        self.label = label


class Cbnz(Op):
    "Conditional branch on non-zero i.e. 1."

    def __init__(self, label):
        self.label = label
