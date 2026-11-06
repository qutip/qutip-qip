import numpy as np
import qutip
import warnings
from abc import ABC, abstractmethod

from qutip import basis, Qobj
from qutip.measurement import measurement_statistics
from qutip_qip.operations import expand_operator
from .op import Op

__all__ = ["Mz", "Mx", "My"]


class Measurement(Op, ABC):
    """
    Base class for quantum measurements.
    """

    name = "M"
    num_qreg: int = 1  # TODO: Make this abstract
    num_creg: int = 1
    is_unitary: bool = False

    @classmethod
    def measurement_comp_basis(cls, state, qubits):
        """
        DEPRECATED: Old method for computational basis meaasurement.

        Measures a particular qubit (determined by the target)
        whose ket vector/ density matrix is specified in the
        computational basis and returns collapsed_states and probabilities
        (retains full dimension).

        Parameters
        ----------
        state : ket or oper
                state to be measured on specified by
                ket vector or density matrix
        qubits : list or tuple of int
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
        warnings.warn(
            "'measurement_comp_basis' has been deprecated and will be removed "
            "in future versions. Please use 'get_measurement_ops()' combined "
            "with simulator logic.",
            DeprecationWarning,
            stacklevel=2,
        )
        n = int(np.log2(state.shape[0]))
        target = qubits[0]
        if target < n:
            op0 = basis(2, 0) * basis(2, 0).dag()
            op1 = basis(2, 1) * basis(2, 1).dag()
            measurement_ops = [op0, op1]
        else:
            raise ValueError("target is not valid")

        measurement_ops = [
            expand_operator(op, dims=[2] * n, targets=qubits) for op in measurement_ops
        ]

        measurement_tol = qutip.settings.core["atol"] ** 2
        states, probabilities = measurement_statistics(state, measurement_ops)
        probabilities = [p if p > measurement_tol else 0.0 for p in probabilities]
        states = [
            s if p > measurement_tol else None for s, p in zip(states, probabilities)
        ]
        return states, probabilities

    @abstractmethod
    def get_measurement_ops(): ...


class Mz(Measurement):
    name = "Mz"
    num_qreg = 1

    @staticmethod
    def get_measurement_ops() -> list[Qobj]:
        op0 = basis(2, 0) * basis(2, 0).dag()
        op1 = basis(2, 1) * basis(2, 1).dag()
        return [op0, op1]


class Mx(Measurement):
    name = "Mx"
    num_qreg = 1

    @staticmethod
    def get_measurement_ops() -> list[Qobj]:
        plus = (basis(2, 0) + basis(2, 1)).unit()
        minus = (basis(2, 0) - basis(2, 1)).unit()
        return [plus * plus.dag(), minus * minus.dag()]


class My(Measurement):
    name = "My"
    num_qreg = 1

    @staticmethod
    def get_measurement_ops() -> list[Qobj]:
        plus_y = (basis(2, 0) + 1j * basis(2, 1)).unit()
        minus_y = (basis(2, 0) - 1j * basis(2, 1)).unit()
        return [plus_y * plus_y.dag(), minus_y * minus_y.dag()]
