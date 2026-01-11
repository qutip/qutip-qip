from numpy.typing import ArrayLike
from qutip import Qobj
from qutip_qip.noise import Noise
from qutip_qip.pulse import Pulse


class DecoherenceNoise(Noise):
    """
    The decoherence noise in a processor. It generates lindblad noise
    according to the given collapse operator `c_ops`.

    Parameters
    ----------
    c_ops : :class:`qutip.Qobj` or list
        The Hamiltonian representing the dynamics of the noise.
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits
    coeff: list, optional
        A list of the coefficients for the control Hamiltonians.
    tlist: array_like, optional
        A NumPy array specifies the time of each coefficient.
    all_qubits: bool, optional
        If `c_ops` contains only single qubits collapse operator,
        ``all_qubits=True`` will allow it to be applied to all qubits.

    Attributes
    ----------
    c_ops : :class:`qutip.Qobj` or list of :class:`qutip.Qobj`
        The Hamiltonian representing the dynamics of the noise.
    targets: int or list
        The indices of qubits that are acted on.
    coeff: list
        A list of the coefficients for the control Hamiltonians.
    tlist: array_like
        A NumPy array specifies the time of each coefficient.
    all_qubits: bool
        If `c_ops` contains only single qubits collapse operator,
        ``all_qubits=True`` will allow it to be applied to all qubits.
    """

    def __init__(
        self,
        c_ops: Qobj | list[Qobj],
        targets: int | list[int] | None = None,
        coeff: list[complex] = None,
        tlist: ArrayLike = None,
        all_qubits: bool = False,
    ):
        if isinstance(c_ops, Qobj):
            self.c_ops = [c_ops]
        else:
            self.c_ops = c_ops
        self.coeff = coeff
        self.tlist = tlist
        self.targets = targets

        if all_qubits:
            if not all([c_op.dims == [[2], [2]] for c_op in self.c_ops]):
                raise ValueError(
                    "The operator is not a single qubit operator, "
                    "thus cannot be applied to all qubits"
                )
        self.all_qubits = all_qubits

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
            [2, 2, ..., 2] for a system of qubits.
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
        if systematic_noise is None:
            systematic_noise = Pulse(None, None, label="system")
        N = len(dims)

        # time-independent
        if (self.coeff is None) and (self.tlist is None):
            self.coeff = True

        for c_op in self.c_ops:
            if self.all_qubits:
                for targets in range(N):
                    systematic_noise.add_lindblad_noise(
                        c_op, targets, self.tlist, self.coeff
                    )
            else:
                systematic_noise.add_lindblad_noise(
                    c_op, self.targets, self.tlist, self.coeff
                )

        return pulses, systematic_noise
