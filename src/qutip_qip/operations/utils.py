from collections.abc import Iterable
from itertools import chain
import numbers

from qutip import Qobj, identity, tensor
from scipy.linalg import block_diag
import numpy as np
import qutip


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


def _check_oper_dims(
    oper: Qobj,
    dims: Iterable[int] | None = None,
    targets: Iterable[int] | None = None,
) -> None:
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

    # If operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError(
                f"The operator dims {oper.dims[0]} do not match "
                f"the target dims {targ_dims}."
            )


def _targets_to_list(
    targets: int | Iterable[int],
    oper: Qobj | None = None,
    N: int | None = None,
) -> list[int]:
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
    if not isinstance(targets, Iterable):
        targets = [targets]
    if not all([isinstance(t, numbers.Integral) for t in targets]):
        raise TypeError("targets should be an integer or a list of integer")

    # if targets has correct length
    if oper is not None:
        req_num = len(oper.dims[0])
        if len(targets) != req_num:
            raise ValueError(
                f"The given operator needs {req_num} "
                f"target qubits, but {len(targets)} given."
            )

    # If targets is smaller than N
    if N is not None:
        if not all([t < N for t in targets]):
            raise ValueError(f"Targets must be smaller than N={N}.")
    return targets


def expand_operator(
    oper: Qobj,
    dims: Iterable[int],
    targets: int | Iterable[int] | None = None,
    dtype: str | None = None,
) -> Qobj:
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
    dtype : str, optional
        Data type of the output `Qobj`.


    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded operator acting on a system with desired dimension.

    Examples
    --------
    >>> import qutip
    >>> from qutip_qip.operations import expand_operator
    >>> from qutip_qip.operations.gates import X, CX
    >>> expand_operator(X.get_qobj(), dims=[2,3], targets=[0])
    Quantum object: dims=[[2, 3], [2, 3]], shape=(6, 6), type='oper', dtype=CSR, isherm=True
    Qobj data =
    [[0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]]
    >>> expand_operator(CX.get_qobj(), dims=[2,2,2], targets=[1, 2])
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
    >>> expand_operator(CX.get_qobj(), dims=[2, 2, 2], targets=[2, 0])
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

    if not isinstance(dims, Iterable):
        raise ValueError(f"dims needs to be an interable {not type(dims)}.")

    N = len(dims)
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_oper_dims(oper, dims=dims, targets=targets)

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


def hadamard_transform(N=1):
    """Quantum object representing the N-qubit Hadamard gate.

    Returns
    -------
    q : qobj
        Quantum object representation of the N-qubit Hadamard gate.

    """
    data = [[1, 1], [1, -1]]
    H = Qobj(data) / np.sqrt(2)

    return tensor([H] * N)


def _flatten(lst):
    """
    Helper to flatten lists.
    """

    return [item for sublist in lst for item in sublist]


def _mult_sublists(tensor_list, overall_inds, U, inds):
    """
    Calculate the revised indices and tensor list by multiplying a new unitary
    U applied to inds.

    Parameters
    ----------
    tensor_list : list of Qobj
        List of gates (unitaries) acting on disjoint qubits.

    overall_inds : list of list of int
        List of qubit indices corresponding to each gate in tensor_list.

    U: Qobj
        Unitary to be multiplied with the the unitary specified by tensor_list.

    inds: list of int
        List of qubit indices corresponding to U.

    Returns
    -------
    tensor_list_revised: list of Qobj
        List of gates (unitaries) acting on disjoint qubits incorporating U.

    overall_inds_revised: list of list of int
        List of qubit indices corresponding to each gate in tensor_list_revised.
    """

    tensor_sublist = []
    inds_sublist = []

    tensor_list_revised = []
    overall_inds_revised = []

    for sub_inds, sub_U in zip(overall_inds, tensor_list):
        if len(set(sub_inds).intersection(inds)) > 0:
            tensor_sublist.append(sub_U)
            inds_sublist.append(sub_inds)
        else:
            overall_inds_revised.append(sub_inds)
            tensor_list_revised.append(sub_U)

    inds_sublist = _flatten(inds_sublist)
    U_sublist = tensor(tensor_sublist)

    revised_inds = list(set(inds_sublist).union(set(inds)))
    N = len(revised_inds)

    sorted_positions = sorted(range(N), key=lambda key: revised_inds[key])
    ind_map = {ind: pos for ind, pos in zip(revised_inds, sorted_positions)}

    U_sublist = expand_operator(
        U_sublist, dims=[2] * N, targets=[ind_map[ind] for ind in inds_sublist]
    )
    U = expand_operator(
        U, dims=[2] * N, targets=[ind_map[ind] for ind in inds]
    )

    U_sublist = U * U_sublist
    inds_sublist = revised_inds

    overall_inds_revised.append(inds_sublist)
    tensor_list_revised.append(U_sublist)

    return tensor_list_revised, overall_inds_revised


def _expand_overall(tensor_list, overall_inds):
    """
    Tensor unitaries in tensor list and then use expand_operator to rearrange
    them appropriately according to the indices in overall_inds.
    """

    U_overall = tensor(tensor_list)
    overall_inds = _flatten(overall_inds)

    # Map indices to a contiguous 0...N-1 range to prevent out-of-bounds in expand_operator
    N = len(overall_inds)
    sorted_positions = sorted(range(N), key=lambda key: overall_inds[key])
    ind_map = {ind: pos for ind, pos in zip(overall_inds, sorted_positions)}
    mapped_targets = [ind_map[ind] for ind in overall_inds]

    U_overall = expand_operator(
        U_overall, dims=[2] * len(overall_inds), targets=mapped_targets
    )
    overall_inds = sorted(overall_inds)
    return U_overall, overall_inds


def _gate_sequence_product(U_list, ind_list):
    """
    Calculate the overall unitary matrix for a given list of unitary operations
    that are still of original dimension.

    Parameters
    ----------
    U_list : list of Qobj
        List of gates(unitaries) implementing the quantum circuit.

    ind_list : list of list of int
        List of qubit indices corresponding to each gate in tensor_list.

    Returns
    -------
    U_overall : qobj
        Unitary matrix corresponding to U_list.

    overall_inds : list of int
        List of qubit indices on which U_overall applies.
    """
    if not U_list:
        return None, []

    num_qubits = len(set(chain(*ind_list)))
    sorted_inds = sorted(set(_flatten(ind_list)))
    ind_list = [[sorted_inds.index(ind) for ind in inds] for inds in ind_list]

    U_overall = None
    overall_inds = []
    tensor_list = []

    for i, (U, inds) in enumerate(zip(U_list, ind_list)):
        # when the tensor_list covers the full dimension of the circuit, we
        # expand the tensor_list to a unitary and call gate_sequence_product
        # recursively on the rest of the U_list.
        if len(overall_inds) == 1 and len(overall_inds[0]) == num_qubits:
            # FIXME undefined variable tensor_list
            U_overall, overall_inds = _expand_overall(
                tensor_list, overall_inds
            )
            U_left, rem_inds = _gate_sequence_product(U_list[i:], ind_list[i:])
            U_left = expand_operator(
                U_left, dims=[2] * num_qubits, targets=rem_inds
            )
            return U_left * U_overall, [
                sorted_inds[ind] for ind in overall_inds
            ]

        if U_overall is None:
            U_overall = U
            overall_inds = [ind_list[0]]
            tensor_list = [U_overall]
            continue

        # case where the next unitary interacts on some subset of qubits
        # with the unitaries already in tensor_list.
        elif len(set(_flatten(overall_inds)).intersection(set(inds))) > 0:
            tensor_list, overall_inds = _mult_sublists(
                tensor_list, overall_inds, U, inds
            )

        # case where the next unitary does not interact with any unitary in
        # tensor_list
        else:
            overall_inds.append(inds)
            tensor_list.append(U)

    U_overall, overall_inds = _expand_overall(tensor_list, overall_inds)

    return U_overall, [sorted_inds[ind] for ind in overall_inds]


def _gate_sequence_product_with_expansion(U_list, left_to_right=True):
    """
    Calculate the overall unitary matrix for a given list of unitary
    operations, assuming that all operations have the same dimension.
    This is only for backward compatibility.

    Parameters
    ----------
    U_list : list
        List of gates(unitaries) implementing the quantum circuit.

    left_to_right : Boolean
        Check if multiplication is to be done from left to right.

    Returns
    -------
    U_overall : qobj
        Unitary matrix corresponding to U_list.
    """

    if len(U_list) == 0:
        raise ValueError("Got an empty U_list")

    U_overall = U_list[0]
    for U in U_list[1:]:
        if left_to_right:
            U_overall = U * U_overall
        else:
            U_overall = U_overall * U

    return U_overall


def gate_sequence_product(
    U_list: list[Qobj],
    left_to_right: bool = True,
    inds_list: list[list[int]] | None = None,
    expand: bool = False,
) -> Qobj | tuple[Qobj, list[int]]:
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
    if expand:
        return _gate_sequence_product(U_list, inds_list)
    else:
        return _gate_sequence_product_with_expansion(U_list, left_to_right)
