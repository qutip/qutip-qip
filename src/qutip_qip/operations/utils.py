import warnings
import numbers
from typing import Sequence
from collections.abc import Iterable

import numpy as np
from scipy.linalg import block_diag
import qutip
from qutip import Qobj, identity, tensor


def _check_oper_dims(oper: Qobj, dims=None, targets=None):
    """
    Check if the given operator is valid.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        The quantum object to be checked.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        e.g ``[2, 2, 2, 2, 2]`` for 5 qubits system.
    targets : int or list of int, optional
        The indices of subspace that are acted on.
    """
    # If operator matches N
    if not isinstance(oper, Qobj) or oper.dims[0] != oper.dims[1]:
        raise ValueError(
            "The operator is not an "
            "Qobj with the same input and output dimensions."
        )
    # if operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError(
                f"The operator dims {oper.dims[0]} do not match "
                "the target dims {targ_dims}."
            )


def _targets_to_list(targets, oper=None, N=None):
    """
    transform targets to a list and check validity.

    Parameters
    ----------
    targets : int or list of int
        The indices of subspace that are acted on.
    oper : :class:`qutip.Qobj`, optional
        An operator, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int, optional
        The number of subspace in the system.
    """
    # if targets is a list of integer
    if targets is None:
        targets = list(range(len(oper.dims[0])))
    if not hasattr(targets, "__iter__"):
        targets = [targets]
    if not all([isinstance(t, numbers.Integral) for t in targets]):
        raise TypeError("targets should be an integer or a list of integer")
    # if targets has correct length
    if oper is not None:
        req_num = len(oper.dims[0])
        if len(targets) != req_num:
            raise ValueError(
                f"The given operator needs {req_num} "
                "target qutbis, "
                "but {len(targets)} given."
            )

    # If targets is smaller than N
    if N is not None:
        if not all([t < N for t in targets]):
            raise ValueError("Targets must be smaller than N={}.".format(N))
    return targets


def expand_operator(
    oper: Qobj,
    N: None = None,
    targets: int | list[int] | None = None,
    dims: list[int] | None = None,
    cyclic_permutation: bool = False,
    dtype: str | None = None,
):
    """
    Expand an operator to one that acts on a system with desired dimensions.

    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        An operator that act on the subsystem, has to be an operator and the
        dimension matches the tensored dims Hilbert space
        e.g. oper.dims = ``[[2, 3], [2, 3]]``
    dims : list
        A list of integer for the dimension of each composite system.
        E.g ``[2, 3, 2, 3, 4]``.
    targets : int or list of int
        The indices of subspace that are acted on.
        Permutation can also be realized by changing the orders of the indices.
    N : int
        Deprecated. Number of qubits. Please use `dims`.
    cyclic_permutation : boolean, optional
        Deprecated.
        Expand for all cyclic permutation of the targets.
        E.g. if ``N=3`` and `oper` is a 2-qubit operator,
        the result will be a list of three operators,
        each acting on qubits 0 and 1, 1 and 2, 2 and 0.
    dtype : str, optional
        Data type of the output `Qobj`. Only for qutip version larger than 5.


    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded operator acting on a system with desired dimension.

    Examples
    --------
    >>> from qutip_qip.operations import expand_operator
    >>> from qutip_qip.operations.std import X, CNOT
    >>> import qutip
    >>> expand_operator(X.get_qobj(), dims=[2,3], targets=[0]) # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 3], [2, 3]], shape=(6, 6), type='oper', dtype=CSR, isherm=True
    Qobj data =
    [[0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]]
    >>> expand_operator(CNOT.get_qobj(), dims=[2,2,2], targets=[1, 2]) # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=CSR, isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 1. 0.]]
    >>> expand_operator(CNOT.get_qobj(), dims=[2, 2, 2], targets=[2, 0]) # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape=(8, 8), type='oper', dtype=CSR, isherm=True
    Qobj data =
    [[1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]]
    """
    dtype = dtype or qutip.settings.core["default_dtype"] or qutip.data.CSR
    oper = oper.to(dtype)

    if N is not None:
        warnings.warn(
            "The function expand_operator has been generalized to "
            "arbitrary subsystems instead of only qubit systems."
            "Please use the new signature e.g.\n"
            "expand_operator(oper, dims=[2, 3, 2, 2], targets=2)",
            DeprecationWarning,
            stacklevel=2,
        )

    if dims is not None and N is None:
        if not isinstance(dims, Iterable):
            f"dims needs to be an interable {not type(dims)}."
        N = len(dims)  # backward compatibility

    if dims is None:
        dims = [2] * N
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_oper_dims(oper, dims=dims, targets=targets)

    # Call expand_operator for all cyclic permutation of the targets.
    if cyclic_permutation:
        warnings.warn(
            "cyclic_permutation is deprecated, "
            "please use loop through different targets manually.",
            DeprecationWarning,
            stacklevel=2,
        )
        oper_list = []
        for i in range(N):
            new_targets = np.mod(np.array(targets) + i, N)
            oper_list.append(
                expand_operator(oper, N=N, targets=new_targets, dims=dims)
            )
        return oper_list

    # Generate the correct order for permutation,
    # eg. if N = 5, targets = [3,0], the order is [1,2,3,0,4].
    # If the operator is cnot,
    # this order means that the 3rd qubit controls the 0th qubit.
    new_order = [0] * N
    for i, t in enumerate(targets):
        new_order[t] = i
    # allocate the rest qutbits (not targets) to the empty
    # position in new_order
    rest_pos = [q for q in list(range(N)) if q not in targets]
    rest_qubits = list(range(len(targets), N))
    for i, ind in enumerate(rest_pos):
        new_order[ind] = rest_qubits[i]
    id_list = [identity(dims[i]) for i in rest_pos]
    out = tensor([oper] + id_list).permute(new_order)
    return out.to(dtype)


def gate_sequence_product(
    U_list, left_to_right=True, inds_list=None, expand=False
):
    """
    Calculate the overall unitary matrix for a given list of unitary operations.

    Parameters
    ----------
    U_list: list
        List of gates implementing the quantum circuit.

    left_to_right: Boolean, optional
        Check if multiplication is to be done from left to right.

    inds_list: list of list of int, optional
        If expand=True, list of qubit indices corresponding to U_list
        to which each unitary is applied.

    expand: Boolean, optional
        Check if the list of unitaries need to be expanded to full dimension.

    Returns
    -------
    U_overall : qobj
        Unitary matrix corresponding to U_list.

    overall_inds : list of int, optional
        List of qubit indices on which U_overall applies.
    """
    from qutip_qip.circuit.simulator import (
        gate_sequence_product,
        gate_sequence_product_with_expansion,
    )

    if expand:
        return gate_sequence_product(U_list, inds_list)
    else:
        return gate_sequence_product_with_expansion(U_list, left_to_right)


def controlled_gate_unitary(
    U: Qobj,
    num_controls: int,
    control_value: int,
) -> Qobj:
    """
    Create an N-qubit controlled gate from a single-qubit gate U with the given
    control and target qubits.

    Parameters
    ----------
    U : :class:`qutip.Qobj`
        An arbitrary unitary gate.
    controls : list of int
        The index of the first control qubit.
    targets : list of int
        The index of the target qubit.
    N : int
        The total number of qubits.
    control_value : int
        The decimal value of the controlled qubits that activates the gate U.

    Returns
    -------
    result : qobj
        Quantum object representing the controlled-U gate.
    """
    # Compatibility
    num_targets = len(U.dims[0])

    # First, assume that the last qubit is the target and control qubits are
    # in the increasing order.
    # The control_value is the location of this unitary.
    target_dim = U.shape[0]
    block_matrices = [np.eye(target_dim) for _ in range(2**num_controls)]
    block_matrices[control_value] = U.full()

    result = block_diag(*block_matrices)
    result = Qobj(result, dims=[[2] * (num_controls + num_targets)] * 2)

    # Expand it to N qubits and permute qubits labelling
    return result
