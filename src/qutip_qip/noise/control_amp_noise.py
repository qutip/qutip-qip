from numpy.typing import ArrayLike
from qutip_qip.noise import Noise
from qutip_qip.pulse import Pulse

class ControlAmpNoise(Noise):
    """
    The noise in the amplitude of the control pulse.

    Parameters
    ----------
    coeff: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`qutip.QobjEvo`.
    tlist: array_like, optional
        A NumPy array specifies the time of each coefficient.
    indices: list of int, optional
        The indices of target pulse in the list of pulses.

    Attributes
    ----------
    coeff: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`qutip.QobjEvo`.
    tlist: array_like
        A NumPy array specifies the time of each coefficient.
    indices: list of int
        The indices of target pulse in the list of pulses.

    """

    def __init__(
        self,
        coeff: list[complex],
        tlist: ArrayLike = None,
        indices: list[int] = None
    ):
        self.coeff = coeff
        self.tlist = tlist
        self.indices = indices

    def get_noisy_pulses(
        self,
        dims = None,
        pulses: list[Pulse] = None,
        systematic_noise: Pulse = None
    ):
        if pulses is None:
            pulses = []

        if self.indices is None:
            indices = range(len(pulses))
        else:
            indices = self.indices

        for i in indices:
            pulse = pulses[i]
            if isinstance(self.coeff, (int, float)):
                coeff = pulse.coeff * self.coeff
            else:
                coeff = self.coeff

            if self.tlist is None:
                tlist = pulse.tlist
            else:
                tlist = self.tlist

            pulses[i].add_coherent_noise(
                pulse.qobj, pulse.targets, tlist, coeff
            )

        return pulses, systematic_noise
