from copy import deepcopy

from qutip_qip.noise import DecoherenceNoise, Noise, RelaxationNoise
from qutip_qip.pulse import Pulse


def process_noise(
    pulses: list[Pulse],
    noise_list: list[Noise],
    dims: int | list[int],
    t1: float | list[float] | None = None,
    t2: float | list[float] | None = None,
    device_noise: bool = False,
    spline_kind: str | None = None,
) -> list[Pulse]:
    """
    Apply noise to the input list of pulses. It does not modify the input
    pulse, but return a new one containing the noise.

    Parameters
    ----------
    pulses: list of :class:`.Pulse`
        The input pulses, on which the noise object will be applied.
    noise_list: list of :class:`.Noise`
        A list of noise objects.
    dims: int or list
        Dimension of the system.
        If int, we assume it is the number of qubits in the system.
        If list, it is the dimension of the component systems.
    t1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `N` or a float for all qubits.
    t2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit. A list of size `N` or a float for all qubits.
    device_noise: bool
        If pulse independent noise such as relaxation are included.
        Default is False.
    spline_kind: str, optional
        Type of the coefficient interpolation:
        "step_func" or "cubic".

    Returns
    -------
    noisy_pulses: list of :class:`.Pulse`
        The noisy pulses, including the system noise.
    """
    noise_list = noise_list.copy()
    noisy_pulses = deepcopy(pulses)
    systematic_noise = Pulse(
        None, None, label="systematic_noise", spline_kind=spline_kind
    )

    if (t1 is not None) or (t2 is not None):
        noise_list.append(RelaxationNoise(t1, t2))

    for noise in noise_list:
        if (
            isinstance(noise, (DecoherenceNoise, RelaxationNoise))
            and not device_noise
        ):
            pass
        else:
            noisy_pulses, systematic_noise = noise._apply_noise(
                dims=dims,
                pulses=noisy_pulses,
                systematic_noise=systematic_noise,
            )

    if device_noise:
        return noisy_pulses + [systematic_noise]
    else:
        return noisy_pulses
