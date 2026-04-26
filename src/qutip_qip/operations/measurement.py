import numpy as np
import warnings
import qutip
from qutip import basis
from qutip.measurement import measurement_statistics
from qutip_qip.operations import expand_operator

__all__ = ["Measurement", "Mz"]


class Measurement:
    """
    Representation of a quantum measurement, with its required parameters,
    and target qubits.
    """

    name = "M"

    def __init__(self, name=None, targets=None, index=None, classical_store=None):
        """
        Create a measurement with specified parameters.
        """
        if name is not None:
            warnings.warn(
                "'name' argument in Measurement has been deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        if index is not None:
            raise AttributeError("argument index is no longer supported")

        if targets is not None or classical_store is not None:
            warnings.warn(
                "'targets' and 'classical_store' arguments in Measurement are deprecated. "
                "Please pass these directly to QubitCircuit.add_measurement() method.",
                DeprecationWarning,
                stacklevel=2,
            )

    def measurement_comp_basis(self, state, targets):
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
        targets : list or tuple of int
                The indices of the qubits to be measured.

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
        target = targets[0]
        if target < n:
            op0 = basis(2, 0) * basis(2, 0).dag()
            op1 = basis(2, 1) * basis(2, 1).dag()
            measurement_ops = [op0, op1]
        else:
            raise ValueError("target is not valid")

        measurement_ops = [
            expand_operator(op, dims=[2] * n, targets=targets) for op in measurement_ops
        ]

        measurement_tol = qutip.settings.core["atol"] ** 2
        states, probabilities = measurement_statistics(state, measurement_ops)
        probabilities = [p if p > measurement_tol else 0.0 for p in probabilities]
        states = [
            s if p > measurement_tol else None for s, p in zip(states, probabilities)
        ]
        return states, probabilities

    def __str__(self):
        if self.name:
            return f" Measurement({self.name})"
        return "Measurement"

    def __repr__(self):
        return str(self)


Mz = Measurement()
