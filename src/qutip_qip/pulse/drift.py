from qutip import QobjEvo, Qobj
from qutip_qip.pulse.utils import EvoElement, merge_qobjevo


class Drift:
    """
    Representation of the time-independent drift Hamiltonian.
    Usually its the intrinsic
    evolution of the quantum system that can not be tuned.

    Parameters
    ----------
    qobj : :class:`qutip.Qobj` or list of :class:`qutip.Qobj`, optional
        The drift Hamiltonians.

    Attributes
    ----------
    qobj: list of :class:`qutip.Qobj`
        A list of the the drift Hamiltonians.
    """

    def __init__(self, qobj: Qobj = None):
        if qobj is None:
            self.drift_hamiltonians = []
        elif isinstance(qobj, list):
            self.drift_hamiltonians = qobj
        else:
            self.drift_hamiltonians = [qobj]

    def add_drift(self, qobj: Qobj, targets: list):
        #TODO add the list type in typehint
        """
        Add a Hamiltonian to the drift.

        Parameters
        ----------
        qobj: :class:`qutip.Qobj`
            The collapse operator of the lindblad noise.
        targets: list
            target qubits of the collapse operator
            (or subquantum system of other dimensions).
        """
        self.drift_hamiltonians.append(EvoElement(qobj, targets))

    def get_ideal_qobjevo(self, dims: int | list[int]) -> QobjEvo:
        """
        Get the QobjEvo representation of the drift Hamiltonian.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        ideal_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the drift evolution.
        """
        if not self.drift_hamiltonians:
            self.drift_hamiltonians = [EvoElement(None, None)]
        qu_list = [
            QobjEvo(evo.get_qobj(dims)) for evo in self.drift_hamiltonians
        ]
        return merge_qobjevo(qu_list)

    def get_noisy_qobjevo(
        self, dims: int | list[int]
    ) -> tuple[QobjEvo, list[QobjEvo]]:
        """
        Same as the `get_ideal_qobjevo` method. There is no additional noise
        for the drift evolution.

        Returns
        -------
        noisy_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the ideal evolution and coherent noise.
        c_ops: list of :class:`qutip.QobjEvo`
            Always an empty list for Drift
        """
        return self.get_ideal_qobjevo(dims), []

    def get_full_tlist(self) -> None:
        """
        Drift has no tlist, this is just a place holder to keep it unified
        with :class:`.Pulse`. It returns None.
        """
        return None
