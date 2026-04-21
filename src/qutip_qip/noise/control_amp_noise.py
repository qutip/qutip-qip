from numpy.typing import ArrayLike

from qutip_qip.noise import Noise
from qutip_qip.pulse import Pulse
from qutip_qip.typing import IntSequence, Real, RealSequence


class ControlAmpNoise(Noise):
    """
    The noise in the amplitude of the control pulse.

    Parameters
    ----------
    coeff : Real | RealSequence
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`qutip.QobjEvo`.
    tlist : array_like, optional
        A NumPy array specifies the time of each coefficient.
    indices : IntSequence, optional
        The indices of target pulse in the list of pulses.

    Attributes
    ----------
    coeff : Real | RealSequence
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`qutip.QobjEvo`.
    tlist : array_like or None
        A NumPy array specifies the time of each coefficient.
    indices : IntSequence or None
        The indices of target pulse in the list of pulses.

    """

    def __init__(
        self,
        coeff: Real | RealSequence,
        tlist: ArrayLike | None = None,
        indices: IntSequence | None = None,
    ) -> None:
        self.coeff = coeff
        self.tlist = tlist
        self.indices = indices

    def get_noisy_pulses(
        self,
        dims: list[int] | None = None,
        pulses: list[Pulse] | None = None,
        systematic_noise: Pulse | None = None,
    ) -> tuple[list[Pulse], Pulse]:
        """
        Return the input pulses list with noise added and
        the pulse independent noise in a dummy :class:`.Pulse` object.

        Parameters
        ----------
        dims : list of int, optional
            The dimension of the components system, the default value is
            [2, 2, ..., 2] for qubits system.
        pulses : list of :class:`.Pulse`, optional
            The input pulses. The noise will be added to pulses in this list.
        systematic_noise : :class:`.Pulse`, optional
            The dummy pulse with no ideal control element.

        Returns
        -------
        noisy_pulses : list of :class:`.Pulse`
            Noisy pulses.
        systematic_noise : :class:`.Pulse`
            The dummy pulse representing pulse-independent noise.
        """
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
