"""Generating qubit states."""

__all__ = ["qubit_states", "truncate_to_qubit_state", "expand_qubit_state"]

import qutip
from qutip import tensor, basis
import numpy as np
from numpy import sqrt


def qubit_states(states):
    """
    Shortcut to generate disentangled qubit states.

    Parameters
    ----------
    states : list or str
        - If a list consisting of ``0``, ``1``, ``"0"``, ``"1"``, ``"+"``
          and ``"-"``, return the corresponding zero/one/plus/minus state.
        - If a string consisting of ``0``, ``1``, ``+``, ``-``,
          same as above.
        - If a list of float or complex numbers,
          each number is mapped to a state of the form
          :math:`\\sqrt{1 - |a|^2} \\left|0\\right\\rangle + a |1\\rangle`,
          where :math:`a` is the given number.

    Returns
    -------
    quantum_states : :obj:`qutip.Qobj`
        The generated qubit states.

    Examples
    --------
    >>> from qutip_qip.qubits import qubit_states
    >>> qubit_states([0, 0])  # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
    Qobj data =
    [[1.]
     [0.]
     [0.]
     [0.]]
    >>> qubit_states([1, "+"])  # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims = [[2, 2], [1, 1]], shape =
    (4, 1), type = ket
    Qobj data =
    [[0.        ]
     [0.        ]
     [0.70710678]
     [0.70710678]]
    >>> qubit_states("-")  # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[ 0.70710678]
     [-0.70710678]]
    >>> qubit_states("1-")  # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims = [[2, 2], [1, 1]], shape =
    (4, 1), type = ket
    Qobj data =
    [[ 0.        ]
     [ 0.        ]
     [ 0.70710678]
     [-0.70710678]]
    >>> import numpy as np
    >>> qubit_states([1.j/np.sqrt(2)])  # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[0.70710678+0.j        ]
     [0.        +0.70710678j]]
    """
    states_map = {
        0: qutip.basis(2, 0),
        1: qutip.basis(2, 1),
        "0": qutip.basis(2, 0),
        "1": qutip.basis(2, 1),
        "+": (qutip.basis(2, 0) + qutip.basis(2, 1)).unit(),
        "-": (qutip.basis(2, 0) - qutip.basis(2, 1)).unit(),
    }

    states_list = []
    for s in states:
        if s in states_map:
            states_list.append(states_map[s])
        elif np.isscalar(s) and abs(s) <= 1:
            states_list.append(
                s * qutip.basis(2, 1)
                + np.sqrt(1 - abs(s) ** 2) * qutip.basis(2, 0)
            )
        else:
            raise TypeError(f"Invalid input {s}.")
    return qutip.tensor(states_list)


def _find_reduced_indices(dims):
    """Find the forresponding indices for a given qu"""
    if len(dims) == 1:
        return np.array([0, 1], dtype=np.intp)
    else:
        rest_reduced_indices = _find_reduced_indices(dims[1:])
        rest_full_dims = np.product(dims[1:])
        #  [indices + 0, indices + a]
        reduced_indices = np.array(
            [[0], [rest_full_dims]], dtype=np.intp
        ) + np.array(
            [rest_reduced_indices, rest_reduced_indices], dtype=np.intp
        )
        return reduced_indices.reshape(reduced_indices.size)


def truncate_to_qubit_state(state):
    """
    Truncate a given quantum state into the computational subspace,
    i.e., each subspace is a 2-level system such as
    ``dims=[[2, 2, 2...],[2, 2, 2...]]``.
    The non-computational state will be discarded.
    Notice that the trace truncated state is in general small than 1.

    Parameters
    ----------
    state : :obj:`qutip.Qobj`
        The input quantum state, either a ket, a bra or a square operator.

    Returns
    -------
    truncated_state : :obj:`qutip.Qobj`
        The truncated state.

    Examples
    --------
    >>> import qutip
    >>> from qutip_qip.qubits import truncate_to_qubit_state
    >>> state = qutip.rand_ket(6, dims=[[2, 3], [1, 1]], seed=0)
    >>> state  # doctest: +NORMALIZE_WHITESPACE
        Quantum object: dims = [[2, 3], [1, 1]], shape = (6, 1), type = ket
        Qobj data =
        [[-0.10369056+0.02570495j]
        [ 0.34852171+0.06053252j]
        [-0.05552249+0.37861125j]
        [ 0.41247492-0.38160681j]
        [ 0.25951892-0.36729024j]
        [ 0.12978757-0.42681426j]]
    >>> truncate_to_qubit_state(state)  # doctest: +NORMALIZE_WHITESPACE
        Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
        Qobj data =
        [[-0.10369056+0.02570495j]
        [ 0.34852171+0.06053252j]
        [ 0.41247492-0.38160681j]
        [ 0.25951892-0.36729024j]]
    """

    if state.isbra:
        dims = state.dims[1]
        reduced_dims = [[1] * len(dims), [2] * len(dims)]
    elif state.isket:
        dims = state.dims[0]
        reduced_dims = [[2] * len(dims), [1] * len(dims)]
    elif state.isoper and state.dims[0] == state.dims[1]:
        dims = state.dims[0]
        reduced_dims = [[2] * len(dims), [2] * len(dims)]
    else:
        raise ValueError("Can only truncate bra, ket or square operator.")
    reduced_indices = _find_reduced_indices(dims)
    # See NumPy Advanced Indexing. Choose the corresponding rows and columns.
    zero_slice = np.array([0], dtype=np.intp)
    reduced_indices1 = reduced_indices if not state.isbra else zero_slice
    reduced_indices2 = reduced_indices if not state.isket else zero_slice
    output = state[reduced_indices1[:, np.newaxis], reduced_indices2]
    return qutip.Qobj(output, dims=reduced_dims)


def expand_qubit_state(state, dims):
    """
    Expand a given qubit state into the a quantum state into a
    multi-dimensional quantum state.
    The additional matrix entries are filled with zeros.

    Parameters
    ----------
    state : :obj:`qutip.Qobj`
        The input quantum state, either a ket, a bra or a square operator.

    Returns
    -------
    expanded_state : :obj:`qutip.Qobj`
        The expanded state.

    Examples
    --------
    >>> import qutip
    >>> from qutip_qip.qubits import expand_qubit_state
    >>> state = qutip.rand_ket(4, dims=[[2, 2], [1, 1]], seed=0)
    >>> state  # doctest: +NORMALIZE_WHITESPACE
        Quantum object: dims = [[2, 2], [1, 1]], shape = (4, 1), type = ket
        Qobj data =
        [[ 0.22362331-0.09566496j]
        [-0.11702026+0.60050111j]
        [ 0.15751346+0.71069216j]
        [ 0.06879596-0.1786583j ]]
    >>> expand_qubit_state(state, [3, 2])  # doctest: +NORMALIZE_WHITESPACE
        Quantum object: dims = [[3, 2], [1, 1]], shape = (6, 1), type = ket
        Qobj data =
        [[ 0.22362331-0.09566496j]
        [-0.11702026+0.60050111j]
        [ 0.15751346+0.71069216j]
        [ 0.06879596-0.1786583j ]
        [ 0.        +0.j        ]
        [ 0.        +0.j        ]]
    """
    if state.isbra:
        reduced_dims = state.dims[1]
        full_dims = [[1] * len(dims), dims]
    elif state.isket:
        reduced_dims = state.dims[0]
        full_dims = [dims, [1] * len(dims)]
    elif state.isoper and state.dims[0] == state.dims[1]:
        reduced_dims = state.dims[0]
        full_dims = [dims, dims]
    else:
        raise ValueError("Can only expand bra, ket or square operator.")
    if not all([d == 2 for d in reduced_dims]):
        raise ValueError("The input state is not a qubit state.")
    reduced_indices = _find_reduced_indices(dims)

    zero_slice = np.array([0], dtype=np.intp)
    output = np.zeros([np.product(d) for d in full_dims], dtype=complex)
    reduced_indices1 = reduced_indices if not state.isbra else zero_slice
    reduced_indices2 = reduced_indices if not state.isket else zero_slice
    output[reduced_indices1[:, np.newaxis], reduced_indices2] = state.full()
    return qutip.Qobj(output, dims=full_dims)
