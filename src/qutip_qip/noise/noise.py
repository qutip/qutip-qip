import warnings
from qutip_qip.pulse import Pulse


class Noise(object):
    """
    The base class representing noise in a processor.
    The noise object can be added to :class:`.device.Processor` and
    contributes to evolution.
    """

    def get_noisy_pulses(
        self,
        dims: list[int] | None = None,
        pulses: list[Pulse] | None = None,
        systematic_noise: Pulse | None = None
    ) -> tuple[list[Pulse], Pulse]:
        """
        Return the input pulses list with noise added and
        the pulse independent noise in a dummy :class:`.Pulse` object.
        This is a template method, a method with the same name and signatures
        needs to be defined in the subclasses.

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
        get_noisy_dynamics = getattr(self, "get_noisy_dynamics", None)

        if get_noisy_dynamics is not None:
            warnings.warn(
                "Using get_noisy_dynamics as the hook function for custom "
                "noise will be deprecated, "
                "please use get_noisy_pulses instead.",
                PendingDeprecationWarning,
            )
            return self.get_noisy_dynamics(dims, pulses, systematic_noise)

        raise NotImplementedError(
            "Subclass error needs a method"
            "`get_noisy_pulses` to process the noise."
        )

    def _apply_noise(
        self,
        dims: list[int] | None = None,
        pulses: list[Pulse] | None = None,
        systematic_noise: Pulse | None = None
    ) -> tuple[list[Pulse], Pulse]:
        """
        For backward compatibility, in case the method has no return value
        or only return the pulse.
        """
        result = self.get_noisy_pulses(
            pulses=pulses, systematic_noise=systematic_noise, dims=dims
        )

        if result is None:  # in-place change
            pass
        elif isinstance(result, tuple) and len(result) == 2:
            pulses, systematic_noise = result
        
        # only pulse
        elif isinstance(result, list) and len(result) == len(pulses):
            pulses = result
        else:
            raise TypeError(
                "Returned value of get_noisy_pulses not understood."
            )
        
        return pulses, systematic_noise
