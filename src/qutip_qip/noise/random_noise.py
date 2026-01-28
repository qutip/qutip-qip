import numpy as np
from qutip_qip.noise import ControlAmpNoise
from qutip_qip.pulse import Pulse


class RandomNoise(ControlAmpNoise):
    """
    Random noise in the amplitude of the control pulse. The arguments for
    the random generator need to be given as key word arguments.

    Parameters
    ----------
    dt: float, optional
        The time interval between two random amplitude. The coefficients
        of the noise are the same within this time range.
    rand_gen: numpy.random, optional
        A random generator in numpy.random, it has to take a ``size``
        parameter as the size of random numbers in the output array.
    indices: list of int, optional
        The indices of target pulse in the list of pulses.
    **kwargs:
        Key word arguments for the random number generator.

    Attributes
    ----------
    dt: float
        The time interval between two random amplitude. The coefficients
        of the noise are the same within this time range.
    rand_gen: numpy.random, optional
        A random generator in numpy.random, it has to take a ``size``
        parameter.
    indices: list of int
        The indices of target pulse in the list of pulses.
    **kwargs:
        Key word arguments for the random number generator.

    Examples
    --------
    >>> gaussnoise = RandomNoise( \
            dt=0.1, rand_gen=np.random.normal, loc=mean, scale=std) \
            # doctest: +SKIP
    """

    def __init__(
        self,
        dt: float,
        rand_gen,  # FIXME add the typing for it (Use Generator instead)
        indices: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(coeff=None, tlist=None)
        self.rand_gen = rand_gen
        self.kwargs = kwargs
        if "size" in kwargs:
            raise ValueError("size is predetermined inside the noise object.")
        self.dt = dt
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
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.
        pulses : list of :class:`.Pulse`
            The input pulses. The noise will be added to pulses in this list.
        systematic_noise : :class:`.Pulse`
            The dummy pulse with no ideal control element.

        Returns
        -------
        noisy_pulses: list of :class:`.Pulse`
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

        t_max = -np.inf
        t_min = np.inf
        for pulse in pulses:
            t_max = max(max(pulse.tlist), t_max)
            t_min = min(min(pulse.tlist), t_min)

        # create new tlist and random coeff
        num_rand = int(np.floor((t_max - t_min) / self.dt)) + 1
        tlist = np.arange(0, self.dt * num_rand, self.dt)[:num_rand] + t_min
        # [:num_rand] for round off error like 0.2*6=1.2000000000002

        for i in indices:
            pulse = pulses[i]
            coeff = self.rand_gen(**self.kwargs, size=num_rand)
            pulses[i].add_coherent_noise(
                pulse.qobj, pulse.targets, tlist, coeff
            )

        return pulses, systematic_noise
