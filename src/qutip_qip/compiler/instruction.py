from copy import deepcopy
import numpy as np


class Instruction:
    """
    Representation of pulses that implement a quantum gate.
    It contains the control pulse required to implement the gate
    on a particular hardware model.

    Parameters
    ----------
    gate: :class:`~.operations.Gate`
        The logical quantum gate (e.g., RX, CNOT) associated with this
        instruction.
    duration: float, optional
        The total execution time for the instruction. If not provided,
        it is derived from the last element of the tlist.
    tlist: array_like, optional
        A list of time points at which the time-dependent coefficients are
        applied. For the default piecewise constant (PWC) pulses used by
        most compilers, len(tlist) = len(coeffs) + 1.
        See :class:`.Pulse` for spline-based alternatives.
    pulse_info: list, optional
        A list of tuples, each tuple corresponding to a pair of pulse label
        and pulse coefficient, in the format (str, array_like).
        This pulses will implement the desired gate.

    Attributes
    ----------
    targets: list, optional
        The target qubits.
    controls: list, optional
        The control qubits.
    used_qubits: set
        Union of the control and target qubits.
    """

    def __init__(self, gate, tlist=None, pulse_info=(), duration=1):
        self.gate = deepcopy(gate)
        self.used_qubits = set()
        if self.targets is not None:
            self.targets.sort()  # Used when comparing the instructions
            self.used_qubits |= set(self.targets)
        if self.controls is not None:
            self.controls.sort()
            self.used_qubits |= set(self.controls)
        self.tlist = tlist
        if self.tlist is not None:
            if np.isscalar(self.tlist):
                self.duration = self.tlist
            elif abs(self.tlist[0]) > 1.0e-8:
                raise ValueError("Pulse time sequence must start from 0")
            else:
                self.duration = self.tlist[-1]
        else:
            self.duration = duration
        self.pulse_info = pulse_info

    @property
    def name(self):
        """
        Corresponding gate name
        """
        return self.gate.name

    @property
    def targets(self):
        """
        Target qubits

        :type: list
        """
        return self.gate.targets

    @property
    def controls(self):
        """
        Control qubits

        :type: list
        """
        return self.gate.controls
