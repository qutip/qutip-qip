from collections import Counter
import random
import numpy as np

from qutip import Qobj
from qiskit.result import Counts


def get_probabilities(state: Qobj) -> np.ndarray:
    """
    Given a state, return an array of corresponding probabilities.

    Parameters
    ----------
    state: Qobj
        Qobj (type - density matrix, ket) state
        obtained after circuit application.

    Returns
    -------
    :class:`np.ndarray`
        Returns the ``numpy`` corresponding to the basis state.
    """
    # Diagonal elements of a density matrix are the probabilities
    if state.type == "oper":
        return state.diag()

    # Squares of coefficients are the probabilities for a ket vector
    return np.square(np.real(state.data_as("ndarray", copy=False)))


def sample_shots(count_probs: dict, shots: int) -> Counts:
    """
    Sample measurements from a given probability distribution.

    Parameters
    ----------
    count_probs: dict
        Probability distribution corresponding
        to different classical outputs.

    shots: int
        Number of shots for emperical estimation.

    Returns
    -------
    :class:`qiskit.result.Counts`
        Returns the ``Counts`` object sampled according to
        the given probabilities and configured shots.
    """
    weights: list[float] = []
    for p in count_probs.values():
        if hasattr(p, "item"):
            weights.append(float(p.item()))  # For multiple choice
        else:
            weights.append(float(p))  # For a trivial circuit with output 1

    samples = random.choices(list(count_probs.keys()), weights, k=shots)
    return Counts(Counter(samples))
