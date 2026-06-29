from qutip_qip.operations import Op


class Label(Op):
    """A static marker in the instruction list."""

    def __init__(self, name):
        self.name = name
