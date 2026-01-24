from copy import deepcopy
from typing import List, Tuple, Hashable

from qutip import Qobj
from qutip_qip.noise import Noise


class Model:
    """
    Template class for a physical model representing quantum hardware.
    The concrete model class does not have to inherit from this,
    as long as the following methods are defined.

    Parameters
    ----------
    num_qubits: int
        The number of qubits.
    dims : list, optional
        The dimension of each component system.
        Default value is a qubit system of ``dim=[2,2,2,...,2]``.
    **params :
        Hardware parameters for the model.

    Attributes
    ----------
    num_qubits: int
        The number of qubits.
    dims : list, optional
        The dimension of each component system.
    params : dict
        Hardware parameters for the model.
    """

    def __init__(self, num_qubits, dims=None, **params):
        self.num_qubits = num_qubits
        self.dims = dims if dims is not None else num_qubits * [2]
        self.params = deepcopy(params)
        self._controls = {}
        self._drift = []
        self._noise = []

    # TODO make this a property
    def get_all_drift(self) -> List[Tuple[Qobj, List[int]]]:
        """
        Get all the drift Hamiltonians.

        Returns
        -------
        drift_hamiltonian_list : list
            A list of drift Hamiltonians in the form of
            ``[(qobj, targets), ...]``.
        """
        return self._drift

    def get_control(self, label: Hashable) -> Tuple[Qobj, List[int]]:
        """
        Get the control Hamiltonian corresponding to the label.

        Parameters
        ----------
        label : hashable object
            A label that identifies the Hamiltonian.

        Returns
        -------
        control_hamiltonian : tuple
            The control Hamiltonian in the form of ``(qobj, targets)``.
        """
        if hasattr(self, "_old_index_label_map"):
            if isinstance(label, int):
                label = self._old_index_label_map[label]
        return self._controls[label]

    def get_control_labels(self) -> List[Hashable]:
        """
        Get a list of all available control Hamiltonians.
        Optional, required only when plotting the pulses or
        using the optimal control algorithm.

        Returns
        -------
        label_list : list of hashable objects
            A list of hashable objects each corresponds to
            an available control Hamiltonian.
        """
        return list(self._controls.keys())

    def get_noise(self) -> List[Noise]:
        """
        Get a list of :obj:`.Noise` objects.
        Single qubit relaxation (T1, T2) are not included here.
        Optional method.

        Returns
        -------
        noise_list : list
            A list of :obj:`.Noise`.
        """
        if not hasattr(self, "_noise"):
            return []
        return self._noise

    def _add_drift(self, qobj, targets):
        if not hasattr(self, "_drift"):
            raise NotImplementedError(
                "The model does not support adding drift."
            )
        self._drift.append((qobj, targets))

    def _add_control(self, label, qobj, targets):
        if not hasattr(self, "_controls"):
            raise NotImplementedError(
                "The model does not support adding controls."
            )
        self._controls[label] = (qobj, targets)

    def _add_noise(self, noise):
        if not hasattr(self, "_noise"):
            raise NotImplementedError(
                "The model does not support adding noise objects."
            )
        self._noise.append(noise)
