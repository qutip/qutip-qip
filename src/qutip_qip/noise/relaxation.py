import numbers
import numpy as np
from collections.abc import Iterable

from qutip import destroy, num
from qutip_qip.noise import Noise
from qutip_qip.pulse import Pulse


class RelaxationNoise(Noise):
    """
    The decoherence on each qubit characterized by two time scales t1 and t2.

    Parameters
    ----------
    t1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits

    Attributes
    ----------
    t1: float or list
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list
        The indices of qubits that are acted on.
    """

    def __init__(
        self,
        t1: float | list[float],
        t2: float | list[float] = None,
        targets: int | list[int] | None = None
    ):
        self.t1 = t1
        self.t2 = t2
        self.targets = targets

    def _T_to_list(self, T: float | list[float], N: int) -> list[float]:
        """
        Check if the relaxation time is valid

        Parameters
        ----------
        T: float or list of float
            The relaxation time
        N: int
            The number of qubits.

        Returns
        -------
        T: list of float
            The relaxation time in Python list form
        """
        if (isinstance(T, numbers.Real) and T > 0) or T is None:
            return [T] * N
        elif isinstance(T, Iterable) and len(T) == N:
            return T
        else:
            raise ValueError(
                "Invalid relaxation time T={},"
                "either the length is not equal to the number of qubits, "
                "or T is not a positive number.".format(T)
            )

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
        pulses : list of :class:`.Pulse`, optional
            The input pulses. The noise will be added to pulses in this list.
        systematic_noise : :class:`.Pulse`, optional
            The dummy pulse with no ideal control element.

        Returns
        -------
        noisy_pulses: list of :class:`.Pulse`
            Noisy pulses.
        systematic_noise : :class:`.Pulse`
            The dummy pulse representing pulse-independent noise.
        """
        if systematic_noise is None:
            systematic_noise = Pulse(None, None, label="system")
        N = len(dims)

        self.t1 = self._T_to_list(self.t1, N)
        self.t2 = self._T_to_list(self.t2, N)
        if len(self.t1) != N or len(self.t2) != N:
            raise ValueError(
                "Length of t1 or t2 does not match N, "
                "len(t1)={}, len(t2)={}".format(len(self.t1), len(self.t2))
            )

        if self.targets is None:
            targets = range(N)
        else:
            targets = self.targets

        for qu_ind in targets:
            t1 = self.t1[qu_ind]
            t2 = self.t2[qu_ind]

            if t1 is not None:
                op = 1 / np.sqrt(t1) * destroy(dims[qu_ind])
                systematic_noise.add_lindblad_noise(op, qu_ind, coeff=True)

            if t2 is not None:
                # Keep the total dephasing ~ exp(-t/t2)
                if t1 is not None:
                    if 2 * t1 < t2:
                        raise ValueError(
                            "t1={}, t2={} does not fulfill "
                            "2*t1>t2".format(t1, t2)
                        )
                    T2_eff = 1.0 / (1.0 / t2 - 1.0 / 2.0 / t1)
                else:
                    T2_eff = t2

                op = 1 / np.sqrt(2 * T2_eff) * 2 * num(dims[qu_ind])
                systematic_noise.add_lindblad_noise(op, qu_ind, coeff=True)

        return pulses, systematic_noise
