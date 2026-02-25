from collections.abc import Iterable
import numbers

import numpy as np

import qutip
from qutip import basis
from qutip.measurement import measurement_statistics
from qutip_qip.operations import expand_operator


class Measurement:
    """
    Representation of a quantum measurement, with its required parameters,
    and target qubits.

    Parameters
    ----------
    name : string
        Measurement name.
    targets : list or int
        Gate targets.
    classical_store : int
        Result of the measurment is stored in this
        classical register of the circuit.
    """

    def __init__(self, name, targets, classical_store=None):
        """
        Create a measurement with specified parameters.
        """

        self.name = name
        self.targets = None
        self.classical_store = classical_store

        if not isinstance(targets, Iterable) and targets is not None:
            self.targets = [targets]
        else:
            self.targets = targets

        for ind_list in [self.targets]:
            if isinstance(ind_list, Iterable):
                all_integer = all(
                    [isinstance(ind, numbers.Integral) for ind in ind_list]
                )
                if not all_integer:
                    raise ValueError("Index of a qubit must be an integer")

    def measurement_comp_basis(self, state):
        """
        Measures a particular qubit (determined by the target)
        whose ket vector/ density matrix is specified in the
        computational basis and returns collapsed_states and probabilities
        (retains full dimension).

        Parameters
        ----------
        state : ket or oper
                state to be measured on specified by
                ket vector or density matrix

        Returns
        -------
        collapsed_states : List of Qobjs
                        the collapsed state obtained after measuring the qubits
                        and obtaining the qubit specified by the target in the
                        state specified by the index.
        probabilities : List of floats
                        the probability of measuring a state in a the state
                        specified by the index.
        """

        n = int(np.log2(state.shape[0]))
        target = self.targets[0]
        if target < n:
            op0 = basis(2, 0) * basis(2, 0).dag()
            op1 = basis(2, 1) * basis(2, 1).dag()
            measurement_ops = [op0, op1]
        else:
            raise ValueError("target is not valid")

        measurement_ops = [
            expand_operator(op, dims=[2] * n, targets=self.targets)
            for op in measurement_ops
        ]

        measurement_tol = qutip.settings.core["atol"] ** 2
        states, probabilities = measurement_statistics(state, measurement_ops)
        probabilities = [
            p if p > measurement_tol else 0.0 for p in probabilities
        ]
        states = [
            s if p > measurement_tol else None
            for s, p in zip(states, probabilities)
        ]
        return states, probabilities

    def __str__(self):
        return f" Measurement({self.name})"

    def __repr__(self):
        return str(self)
