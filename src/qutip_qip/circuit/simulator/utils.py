from itertools import chain
from qutip import tensor
from qutip_qip.operations import expand_operator


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

    Examples
    --------

    First, we get some imports out of the way,

    >>> from qutip_qip.operations.gates import _mult_sublists
    >>> from qutip_qip.operations.gates import x_gate, y_gate, toffoli, z_gate

    Suppose we have a unitary list of already processed gates,
    X, Y, Z applied on qubit indices 0, 1, 2 respectively and
    encounter a new TOFFOLI gate on qubit indices (0, 1, 3).

    >>> tensor_list = [x_gate(), y_gate(), z_gate()]
    >>> overall_inds = [[0], [1], [2]]
    >>> U = toffoli()
    >>> U_inds = [0, 1, 3]

    Then, we can use _mult_sublists to produce a new list of unitaries by
    multiplying TOFFOLI (and expanding) only on the qubit indices involving
    TOFFOLI gate (and any multiplied gates).

    >>> U_list, overall_inds = _mult_sublists(tensor_list, overall_inds, U, U_inds)
    >>> np.testing.assert_allclose(U_list[0]) == z_gate())
    >>> toffoli_xy = toffoli() * tensor(x_gate(), y_gate(), identity(2))
    >>> np.testing.assert_allclose(U_list[1]), toffoli_xy)
    >>> overall_inds = [[2], [0, 1, 3]]
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
    U_overall = expand_operator(
        U_overall, dims=[2] * len(overall_inds), targets=overall_inds
    )
    overall_inds = sorted(overall_inds)
    return U_overall, overall_inds


def gate_sequence_product(U_list, ind_list):
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

    Examples
    --------

    First, we get some imports out of the way,

    >>> from qutip_qip.operations.gates import gate_sequence_product
    >>> from qutip_qip.operations.gates import x_gate, y_gate, toffoli, z_gate

    Suppose we have a circuit with gates X, Y, Z, TOFFOLI
    applied on qubit indices 0, 1, 2 and [0, 1, 3] respectively.

    >>> tensor_lst = [x_gate(), y_gate(), z_gate(), toffoli()]
    >>> overall_inds = [[0], [1], [2], [0, 1, 3]]

    Then, we can use gate_sequence_product to produce a single unitary
    obtained by multiplying unitaries in the list using heuristic methods
    to reduce the size of matrices being multiplied.

    >>> U_list, overall_inds = gate_sequence_product(tensor_lst, overall_inds)
    """
    num_qubits = len(set(chain(*ind_list)))
    sorted_inds = sorted(set(_flatten(ind_list)))
    ind_list = [[sorted_inds.index(ind) for ind in inds] for inds in ind_list]

    U_overall = 1
    overall_inds = []

    for i, (U, inds) in enumerate(zip(U_list, ind_list)):
        # when the tensor_list covers the full dimension of the circuit, we
        # expand the tensor_list to a unitary and call gate_sequence_product
        # recursively on the rest of the U_list.
        if len(overall_inds) == 1 and len(overall_inds[0]) == num_qubits:
            # FIXME undefined variable tensor_list
            U_overall, overall_inds = _expand_overall(
                tensor_list, overall_inds
            )
            U_left, rem_inds = gate_sequence_product(U_list[i:], ind_list[i:])
            U_left = expand_operator(
                U_left, dims=[2] * num_qubits, targets=rem_inds
            )
            return U_left * U_overall, [
                sorted_inds[ind] for ind in overall_inds
            ]

        # special case for first unitary in the list
        if U_overall == 1:
            U_overall = U_overall * U
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


def gate_sequence_product_with_expansion(U_list, left_to_right=True):
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

    U_overall = 1
    for U in U_list:
        if left_to_right:
            U_overall = U * U_overall
        else:
            U_overall = U_overall * U

    return U_overall
