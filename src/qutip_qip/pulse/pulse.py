import numpy as np
from qutip import QobjEvo, Qobj
from qutip_qip.pulse.utils import EvoElement


class Pulse:
    """
    Representation of a control pulse and the pulse dependent noise.
    The pulse is characterized by the ideal control pulse, the coherent
    noise and the lindblad noise. The later two are lists of
    noisy evolution dynamics.
    Each dynamic element is characterized by four variables:
    ``qobj``, ``targets``, ``tlist`` and ``coeff``.

    See examples for different construction behavior.

    Parameters
    ----------
    qobj : :class:`qutip.Qobj`
        The Hamiltonian of the ideal pulse.
    targets: list
        target qubits of the ideal pulse
        (or subquantum system of other dimensions).
    tlist: array-like, optional
        Time sequence of the ideal pulse.
        A list of time at which the time-dependent coefficients are applied.
        ``tlist`` does not have to be equidistant, but must have the same length
        or one element shorter compared to ``coeff``. See documentation for
        the parameter ``spline_kind``.
    coeff: array-like or bool, optional
        Time-dependent coefficients of the ideal control pulse.
        If an array, the length
        must be the same or one element longer compared to ``tlist``.
        See documentation for the parameter ``spline_kind``.
        If a bool, the coefficient is a constant 1 or 0.
    spline_kind: str, optional
        Type of the coefficient interpolation:
        "step_func" or "cubic".

        -"step_func":
        The coefficient will be treated as a step function.
        E.g. ``tlist=[0,1,2]`` and ``coeff=[3,2]``, means that the coefficient
        is 3 in t=[0,1) and 2 in t=[2,3). It requires
        ``len(coeff)=len(tlist)-1`` or ``len(coeff)=len(tlist)``, but
        in the second case the last element of ``coeff`` has no effect.

        -"cubic":
        Use cubic interpolation for the coefficient. It requires
        ``len(coeff)=len(tlist)``
    label: str
        The label (name) of the pulse.

    Attributes
    ----------
    ideal_pulse: :class:`.pulse.EvoElement`
        The ideal dynamic of the control pulse.
    coherent_noise: list of :class:`.pulse.EvoElement`
        The coherent noise caused by the control pulse. Each dynamic element is
        still characterized by a time-dependent Hamiltonian.
    lindblad_noise: list of :class:`.pulse.EvoElement`
        The dissipative noise of the control pulse. Each dynamic element
        will be treated as a (time-dependent) lindblad operator in the
        master equation.
    spline_kind: str
        See parameter ``spline_kind``.
    label: str
        See parameter ``label``.

    Examples
    --------
    Create a pulse that is turned off

    >>> Pulse(sigmaz(), 0) # doctest: +SKIP
    >>> Pulse(sigmaz(), 0, None, None) # doctest: +SKIP

    Create a time dependent pulse

    >>> tlist = np.array([0., 1., 2., 4.]) # doctest: +SKIP
    >>> coeff = np.array([0.5, 1.2, 0.8]) # doctest: +SKIP
    >>> spline_kind = "step_func" # doctest: +SKIP
    >>> Pulse(sigmaz(), 0, tlist=tlist, coeff=coeff, spline_kind="step_func") # doctest: +SKIP

    Create a time independent pulse

    >>> Pulse(sigmaz(), 0, coeff=True) # doctest: +SKIP

    Create a constant pulse with time range

    >>> Pulse(sigmaz(), 0, tlist=tlist, coeff=True) # doctest: +SKIP

    Create an dummy Pulse (H=0)

    >>> Pulse(None, None) # doctest: +SKIP

    """

    def __init__(
        self,
        qobj: Qobj,
        targets: list[int],
        tlist: list[float] = None,
        coeff: list[float] | bool = None,
        spline_kind: str = "step_func",
        label: str = "",
    ):
        self.spline_kind = spline_kind
        self.ideal_pulse = EvoElement(qobj, targets, tlist, coeff)
        self.coherent_noise = []
        self.lindblad_noise = []
        self.label = label

    @property
    def qobj(self) -> Qobj:
        """
        See parameter `qobj`.
        """
        return self.ideal_pulse.qobj

    @qobj.setter
    def qobj(self, x: Qobj):
        self.ideal_pulse.qobj = x

    @property
    def targets(self) -> list[int]:
        """
        See parameter `targets`.
        """
        return self.ideal_pulse.targets

    @targets.setter
    def targets(self, x: list[int]):
        self.ideal_pulse.targets = x

    @property
    def tlist(self) -> list[float]:
        """
        See parameter `tlist`
        """
        return self.ideal_pulse.tlist

    @tlist.setter
    def tlist(self, x: list[float]):
        self.ideal_pulse.tlist = x

    @property
    def coeff(self) -> list[float] | bool:
        """
        See parameter ``coeff``.
        """
        return self.ideal_pulse.coeff

    @coeff.setter
    def coeff(self, x: list[float] | bool):
        self.ideal_pulse.coeff = x

    def add_coherent_noise(
        self,
        qobj: Qobj,
        targets: list[int],
        tlist: list[float] = None,
        coeff: list[float] | bool = None
    ):
        """
        Add a new (time-dependent) Hamiltonian to the coherent noise.

        Parameters
        ----------
        qobj: :class:`qutip.Qobj`
            The Hamiltonian of the pulse.
        targets: list
            target qubits of the pulse
            (or subquantum system of other dimensions).
        tlist: array-like, optional
            A list of time at which the time-dependent coefficients are
            applied.
            ``tlist`` does not have to be equidistant, but must have the same
            length
            or one element shorter compared to ``coeff``. See documentation for
            the parameter ``spline_kind`` of :class:`.Pulse`.
        coeff: array-like or bool, optional
            Time-dependent coefficients of the pulse noise.
            If an array, the length
            must be the same or one element longer compared to ``tlist``.
            See documentation for
            the parameter ``spline_kind`` of :class:`.Pulse`.
            If a bool, the coefficient is a constant 1 or 0.
        """
        self.coherent_noise.append(EvoElement(qobj, targets, tlist, coeff))

    def add_control_noise(
        self,
        qobj: Qobj,
        targets: list[int],
        tlist: list[float] = None,
        coeff: list[float] | bool = None
    ):
        self.add_coherent_noise(qobj, targets, tlist=tlist, coeff=coeff)

    def add_lindblad_noise(
        self,
        qobj: Qobj,
        targets: list[int],
        tlist: list[float] = None,
        coeff: list[float] | bool = None
    ):
        """
        Add a new (time-dependent) lindblad noise to the coherent noise.

        Parameters
        ----------
        qobj: :class:`qutip.Qobj`
            The collapse operator of the lindblad noise.
        targets: list
            target qubits of the collapse operator
            (or subquantum system of other dimensions).
        tlist: array-like, optional
            A list of time at which the time-dependent coefficients are
            applied.
            ``tlist`` does not have to be equidistant, but must have the same
            length
            or one element shorter compared to ``coeff``.
            See documentation for
            the parameter ``spline_kind`` of :class:`.Pulse`.
        coeff: array-like or bool, optional
            Time-dependent coefficients of the pulse noise.
            If an array, the length
            must be the same or one element longer compared to ``tlist``.
            See documentation for
            the parameter ``spline_kind`` of :class:`.Pulse`.
            If a bool, the coefficient is a constant 1 or 0.
        """
        self.lindblad_noise.append(EvoElement(qobj, targets, tlist, coeff))

    def get_ideal_qobj(self, dims: int | list[int]) -> Qobj:
        """
        Get the Hamiltonian of the ideal pulse.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        qobj : :class:`qutip.Qobj`
            The Hamiltonian of the ideal pulse.
        """
        return self.ideal_pulse.get_qobj(dims)

    def get_ideal_qobjevo(self, dims: int | list[int]) -> QobjEvo:
        """
        Get a `QobjEvo` representation of the ideal evolution.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        ideal_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the ideal evolution.
        """
        return self.ideal_pulse.get_qobjevo(self.spline_kind, dims)

    def get_noisy_qobjevo(self, dims: int | list[int]) -> QobjEvo:
        """
        Get the `QobjEvo` representation of the noisy evolution. The result
        can be used directly as input for the qutip solvers.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        noisy_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the ideal evolution and coherent noise.
        c_ops: list of :class:`qutip.QobjEvo`
            A list of (time-dependent) lindbald operators.
        """
        ideal_qu = self.get_ideal_qobjevo(dims)
        noise_qu_list = [
            noise.get_qobjevo(self.spline_kind, dims)
            for noise in self.coherent_noise
        ]
        qu = sum(noise_qu_list, ideal_qu)
        c_ops = [
            noise.get_qobjevo(self.spline_kind, dims)
            for noise in self.lindblad_noise
        ]
        return qu, c_ops

    def get_full_tlist(self, tol: float = 1.0e-10) -> list[list[float]]:
        """
        Return the full tlist of the pulses and noise.
        It means that if different ``tlist`` are present,
        they will be merged
        to one with all time points stored in a sorted array.

        Returns
        -------
        full_tlist: array-like 1d
            The full time sequence for the noisy evolution.
        """
        # TODO add test
        all_tlists = []
        all_tlists.append(self.ideal_pulse.tlist)
        for pulse in self.coherent_noise:
            all_tlists.append(pulse.tlist)
        for c_op in self.lindblad_noise:
            all_tlists.append(c_op.tlist)
        all_tlists = [tlist for tlist in all_tlists if tlist is not None]
        if not all_tlists:
            return None
        full_tlist = np.unique(np.sort(np.hstack(all_tlists)))
        full_tlist = np.concatenate(
            (full_tlist[:1], full_tlist[1:][np.diff(full_tlist) > tol])
        )
        return full_tlist

    def print_info(self):
        """
        Print the information of the pulse, including the ideal dynamics,
        the coherent noise and the lindblad noise.
        """
        print(
            "-----------------------------------"
            "-----------------------------------"
        )
        if self.label is not None:
            print("Pulse label:", self.label)
        print(
            "The pulse contains: {} coherent noise elements and {} "
            "Lindblad noise elements.".format(
                len(self.coherent_noise), len(self.lindblad_noise)
            )
        )
        print()
        print("Ideal pulse:")
        print(self.ideal_pulse)
        if self.coherent_noise:
            print()
            print("Coherent noise:")
            for ele in self.coherent_noise:
                print(ele)
        if self.lindblad_noise:
            print()
            print("Lindblad noise:")
            for ele in self.lindblad_noise:
                print(ele)
        print(
            "-----------------------------------"
            "-----------------------------------"
        )
