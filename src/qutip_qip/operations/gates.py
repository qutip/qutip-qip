"""
Deprecated module, will be removed in future versions.
"""

from itertools import product
from functools import partial, reduce
from operator import mul

import warnings
import numpy as np
import scipy.sparse as sp

from qutip import Qobj, identity, qeye, sigmax, sigmay, sigmaz, tensor, fock_dm
from qutip_qip.operations import expand_operator


# Single Qubit Gates
def _deprecation_warnings_gate_expansion():
    warnings.warn(
        "The expansion of output gate matrix is no longer included "
        "in the gate functions. "
        "To expand the output `Qobj` or permute the qubits, "
        "please use expand_operator.",
        DeprecationWarning,
        stacklevel=2,
    )


def x_gate(N=None, target=0):
    """Pauli-X gate or sigmax operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the x-axis.

    """
    warnings.warn(
        "x_gate has been deprecated and will be removed in future version. \
        Use X.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(x_gate(), dims=[2] * N, targets=target)
    return sigmax()


def y_gate(N=None, target=0):
    """Pauli-Y gate or sigmay operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the y-axis.

    """
    warnings.warn(
        "Y_gate has been deprecated and will be removed in future version. \
        Use Y.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(y_gate(), dims=[2] * N, targets=target)
    return sigmay()


def z_gate(N=None, target=0):
    """Pauli-Z gate or sigmaz operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the z-axis.

    """
    warnings.warn(
        "z_gate has been deprecated and will be removed in future version. \
        Use Z.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(z_gate(), dims=[2] * N, targets=target)
    return sigmaz()


def cy_gate(N=None, control=0, target=1):
    """Controlled Y gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "cy_gate has been deprecated and will be removed in future version. \
        Use CY.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            cy_gate(), dims=[2] * N, targets=(control, target)
        )
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dims=[[2, 2], [2, 2]],
    )


def cz_gate(N=None, control=0, target=1):
    """Controlled Z gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "cz_gate has been deprecated and will be removed in future version. \
        Use CZ.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            cz_gate(), dims=[2] * N, targets=(control, target)
        )
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dims=[[2, 2], [2, 2]],
    )


def s_gate(N=None, target=0):
    """Single-qubit rotation also called Phase gate or the Z90 gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a 90 degree rotation around the z-axis.

    """
    warnings.warn(
        "s_gate has been deprecated and will be removed in future version. \
        Use S.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(s_gate(), dims=[2] * N, targets=target)
    return Qobj([[1, 0], [0, 1j]])


def cs_gate(N=None, control=0, target=1):
    """Controlled S gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "cs_gate has been deprecated and will be removed in future version. \
        Use CS.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            cs_gate(), dims=[2] * N, targets=(control, target)
        )
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]],
        dims=[[2, 2], [2, 2]],
    )


def t_gate(N=None, target=0):
    """Single-qubit rotation related to the S gate by the relationship S=T*T.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing a phase shift of pi/4.

    """
    warnings.warn(
        "t_gate has been deprecated and will be removed in future version. \
        Use T.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(t_gate(), dims=[2] * N, targets=target)
    return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def ct_gate(N=None, control=0, target=1):
    """Controlled T gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "ct_gate has been deprecated and will be removed in future version. \
        Use CT.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            ct_gate(), dims=[2] * N, targets=(control, target)
        )
    return Qobj(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * np.pi / 4)],
        ],
        dims=[[2, 2], [2, 2]],
    )


def rx(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmax with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "rxRTNOT has been deprecated and will be removed in future version. \
        Use RX(angle).get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(rx(phi), dims=[2] * N, targets=target)
    return Qobj(
        [
            [np.cos(phi / 2), -1j * np.sin(phi / 2)],
            [-1j * np.sin(phi / 2), np.cos(phi / 2)],
        ]
    )


def ry(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmay with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "ryRTNOT has been deprecated and will be removed in future version. \
        Use RY(angle).get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(ry(phi), dims=[2] * N, targets=target)
    return Qobj(
        [
            [np.cos(phi / 2), -np.sin(phi / 2)],
            [np.sin(phi / 2), np.cos(phi / 2)],
        ]
    )


def rz(phi, N=None, target=0):
    """Single-qubit rotation for operator sigmaz with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    warnings.warn(
        "rzRTNOT has been deprecated and will be removed in future version. \
        Use RZ(angle).get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(rz(phi), dims=[2] * N, targets=target)
    return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])


def sqrtnot(N=None, target=0):
    """Single-qubit square root NOT gate.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the square root NOT gate.

    """
    warnings.warn(
        "sqrtnot has been deprecated and will be removed in future version. \
        Use SQRTNOT.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(sqrtnot(), dims=[2] * N, targets=target)
    return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])


def snot(N=None, target=0):
    """Quantum object representing the SNOT (Hadamard) gate.

    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT gate.

    Examples
    --------
    >>> snot() # doctest: +SKIP
    Quantum object: dims=[[2], [2]], \
shape = [2, 2], type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 0.70710678+0.j  0.70710678+0.j]
     [ 0.70710678+0.j -0.70710678+0.j]]

    """
    warnings.warn(
        "snot has been deprecated and will be removed in future version. \
        Use H.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(snot(), dims=[2] * N, targets=target)
    return 1 / np.sqrt(2.0) * Qobj([[1, 1], [1, -1]])


def phasegate(theta, N=None, target=0):
    """
    Returns quantum object representing the phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of phase shift gate.

    Examples
    --------
    >>> phasegate(pi/4) # doctest: +SKIP
    Quantum object: dims=[[2], [2]], \
shape = [2, 2], type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.00000000+0.j          0.00000000+0.j        ]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    warnings.warn(
        "phase has been deprecated and will be removed in future version. \
        Use PHASE(angle).get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(phasegate(theta), dims=[2] * N, targets=target)
    return Qobj([[1, 0], [0, np.exp(1.0j * theta)]], dims=[[2], [2]])


def qrot(theta, phi, N=None, target=0):
    """
    Single qubit rotation driving by Rabi oscillation with 0 detune.

    Parameters
    ----------
    phi : float
        The inital phase of the rabi pulse.
    theta : float
        The duration of the rabi pulse.
    N : int
        Number of qubits in the system.
    target : int
        The index of the target qubit.

    Returns
    -------
    qrot_gate : :class:`qutip.Qobj`
        Quantum object representation of physical qubit rotation under
        a rabi pulse.
    """
    warnings.warn(
        "qrot has been deprecated and will be removed in future version. \
        Use R([theta, phi]).get_qobj() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(qrot(theta, phi), dims=[2] * N, targets=target)
    return Qobj(
        [
            [
                np.cos(theta / 2.0),
                -1.0j * np.exp(-1.0j * phi) * np.sin(theta / 2.0),
            ],
            [
                -1.0j * np.exp(1.0j * phi) * np.sin(theta / 2.0),
                np.cos(theta / 2.0),
            ],
        ]
    )


def qasmu_gate(args, N=None, target=0):
    """
    QASM U-gate as defined in the OpenQASM standard.

    Parameters
    ----------

    theta : float
        The argument supplied to the last RZ rotation.
    phi : float
        The argument supplied to the middle RY rotation.
    gamma : float
        The argument supplied to the first RZ rotation.
    N : int
        Number of qubits in the system.
    target : int
        The index of the target qubit.

    Returns
    -------
    qasmu_gate : :class:`qutip.Qobj`
        Quantum object representation of the QASM U-gate as defined in the
        OpenQASM standard.
    """
    warnings.warn(
        "qasmu_gate has been deprecated and will be removed in future version. \
        Use QASMU([theta, phi, gamma]).get_qobj() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    theta, phi, gamma = args
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            qasmu_gate([theta, phi, gamma]), dims=[2] * N, targets=target
        )
    return Qobj(rz(phi) * ry(theta) * rz(gamma))


#
# 2 Qubit Gates
#


def cphase(theta, N=2, control=0, target=1):
    """
    Returns quantum object representing the controlled phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    N : integer
        The number of qubits in the target space.

    control : integer
        The index of the control qubit.

    target : integer
        The index of the target qubit.

    Returns
    -------
    U : qobj
        Quantum object representation of controlled phase gate.
    """
    warnings.warn(
        "cphase has been deprecated and will be removed in future version. \
        Use CPHASE(angle).get_qobj() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if N != 2 or control != 0 or target != 1:
        _deprecation_warnings_gate_expansion()

    if N < 1 or target < 0 or control < 0:
        raise ValueError("Minimum value: N=1, control=0 and target=0")

    if control >= N or target >= N:
        raise ValueError("control and target need to be smaller than N")

    U_list1 = [identity(2)] * N
    U_list2 = [identity(2)] * N

    U_list1[control] = fock_dm(2, 1)
    U_list1[target] = phasegate(theta)

    U_list2[control] = fock_dm(2, 0)

    U = tensor(U_list1) + tensor(U_list2)
    return U


def cnot(N=None, control=0, target=1):
    """
    Quantum object representing the CNOT gate.

    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate

    Examples
    --------
    >>> cnot() # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]

    """
    warnings.warn(
        "cnot has been deprecated and will be removed in future version. \
        Use CX.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(cnot(), dims=[2] * N, targets=(control, target))
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dims=[[2, 2], [2, 2]],
    )


def csign(N=None, control=0, target=1):
    """
    Quantum object representing the CSIGN gate.

    Returns
    -------
    csign_gate : qobj
        Quantum object representation of CSIGN gate

    Examples
    --------
    >>> csign() # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  -1.+0.j]]

    """
    warnings.warn(
        "csign has been deprecated and will be removed in future version. \
        Use CZ.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )

    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(csign(), N, (control, target))
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dims=[[2, 2], [2, 2]],
    )


def berkeley(N=None, targets=[0, 1]):
    """
    Quantum object representing the Berkeley gate.

    Returns
    -------
    berkeley_gate : qobj
        Quantum object representation of Berkeley gate

    Examples
    --------
    >>> berkeley() # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=True
    Qobj data =
        [[ cos(pi/8).+0.j  0.+0.j           0.+0.j           0.+sin(pi/8).j]
         [ 0.+0.j          cos(3pi/8).+0.j  0.+sin(3pi/8).j  0.+0.j]
         [ 0.+0.j          0.+sin(3pi/8).j  cos(3pi/8).+0.j  0.+0.j]
         [ 0.+sin(pi/8).j  0.+0.j           0.+0.j           cos(pi/8).+0.j]]

    """
    warnings.warn(
        "berkley has been deprecated and will be removed in future version. \
        Use BERKELEY.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )

    if (targets[0] == 1 and targets[1] == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(berkeley(), N, targets=targets)
    return Qobj(
        [
            [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
            [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
            [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
            [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
        ],
        dims=[[2, 2], [2, 2]],
    )


def swapalpha(alpha, N=None, targets=[0, 1]):
    """
    Quantum object representing the SWAPalpha gate.

    Returns
    -------
    swapalpha_gate : qobj
        Quantum object representation of SWAPalpha gate

    Examples
    --------
    >>> swapalpha(alpha) # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0.j  0.+0.j                    0.+0.j                    0.+0.j]
     [ 0.+0.j  0.5*(1 + exp(j*pi*alpha)  0.5*(1 - exp(j*pi*alpha)  0.+0.j]
     [ 0.+0.j  0.5*(1 - exp(j*pi*alpha)  0.5*(1 + exp(j*pi*alpha)  0.+0.j]
     [ 0.+0.j  0.+0.j                    0.+0.j                    1.+0.j]]

    """
    warnings.warn(
        "swapalpha has been deprecated and will be removed in future version. \
        Use SWAPALPHA(angle).get_qobj() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if (targets[0] == 1 and targets[1] == 0) and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(swapalpha(alpha), N, targets=targets)
    return Qobj(
        [
            [1, 0, 0, 0],
            [
                0,
                0.5 * (1 + np.exp(1.0j * np.pi * alpha)),
                0.5 * (1 - np.exp(1.0j * np.pi * alpha)),
                0,
            ],
            [
                0,
                0.5 * (1 - np.exp(1.0j * np.pi * alpha)),
                0.5 * (1 + np.exp(1.0j * np.pi * alpha)),
                0,
            ],
            [0, 0, 0, 1],
        ],
        dims=[[2, 2], [2, 2]],
    )


def swap(N=None, targets=[0, 1]):
    """Quantum object representing the SWAP gate.

    Returns
    -------
    swap_gate : qobj
        Quantum object representation of SWAP gate

    Examples
    --------
    >>> swap() # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=True
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    warnings.warn(
        "SWAP has been deprecated and will be removed in future version. \
        Use SWAP.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(swap(), dims=[2] * N, targets=targets)
    return Qobj(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]],
    )


def iswap(N=None, targets=[0, 1]):
    """Quantum object representing the iSWAP gate.

    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate

    Examples
    --------
    >>> iswap() # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
     [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]]
    """
    warnings.warn(
        "ISWAP has been deprecated and will be removed in future version. \
        Use ISWAP.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(iswap(), dims=[2] * N, targets=targets)
    return Qobj(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]],
    )


def sqrtswap(N=None, targets=[0, 1]):
    """Quantum object representing the square root SWAP gate.

    Returns
    -------
    sqrtswap_gate : qobj
        Quantum object representation of square root SWAP gate

    """
    warnings.warn(
        "SQRTSWAP has been deprecated and will be removed in future version. \
        Use SQRTSWAP.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(sqrtswap(), dims=[2] * N, targets=targets)
    return Qobj(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                [0, 0, 0, 1],
            ]
        ),
        dims=[[2, 2], [2, 2]],
    )


def sqrtiswap(N=None, targets=[0, 1]):
    """Quantum object representing the square root iSWAP gate.

    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate

    Examples
    --------
    >>> sqrtiswap() # doctest: +SKIP
    Quantum object: dims=[[2, 2], [2, 2]], \
shape = [4, 4], type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 1.00000000+0.j   0.00000000+0.j   \
       0.00000000+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.70710678+0.j   \
       0.00000000-0.70710678j  0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000-0.70710678j\
       0.70710678+0.j          0.00000000+0.j]
     [ 0.00000000+0.j   0.00000000+0.j   \
       0.00000000+0.j          1.00000000+0.j]]

    """
    warnings.warn(
        "SQRTISWAP has been deprecated and will be removed in future version. \
        Use SQRTISWAP.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(sqrtiswap(), N, targets=targets)
    return Qobj(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        ),
        dims=[[2, 2], [2, 2]],
    )


def molmer_sorensen(theta, phi=0.0, N=None, targets=[0, 1]):
    """
    Quantum object of a Mølmer–Sørensen gate.

    Parameters
    ----------
    theta: float
        The duration of the interaction pulse.
    phi: float
        Rotation axis. phi = 0 for XX; phi=pi for YY
    N: int
        Number of qubits in the system.
    target: int
        The indices of the target qubits.

    Returns
    -------
    molmer_sorensen_gate : :class:`qutip.Qobj`
        Quantum object representation of the Mølmer–Sørensen gate.
    """
    warnings.warn(
        "MS has been deprecated and will be removed in future version. \
        Use MS([theta, phi]).get_qobj() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if targets != [0, 1] and N is None:
        N = 2

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            molmer_sorensen(theta, phi), dims=[2] * N, targets=targets
        )

    return Qobj(
        [
            [
                np.cos(theta / 2),
                0,
                0,
                -1j * np.exp(-1j * 2 * phi) * np.sin(theta / 2),
            ],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [
                -1j * np.exp(1j * 2 * phi) * np.sin(theta / 2),
                0,
                0,
                np.cos(theta / 2),
            ],
        ],
        dims=[[2, 2], [2, 2]],
    )


#
# 3 Qubit Gates
#


def fredkin(N=None, control=0, targets=[1, 2]):
    """Quantum object representing the Fredkin gate.

    Returns
    -------
    fredkin_gate : qobj
        Quantum object representation of Fredkin gate.

    Examples
    --------
    >>> fredkin() # doctest: +SKIP
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape = [8, 8], type='oper', dtype=Dense, isherm=True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]]

    """
    warnings.warn(
        "fredkin has been deprecated and will be removed in future version. \
        Use FREDKIN.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if [control, targets[0], targets[1]] != [0, 1, 2] and N is None:
        N = 3

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            fredkin(), dims=[2] * N, targets=(control,) + tuple(targets)
        )
    return Qobj(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dims=[[2, 2, 2], [2, 2, 2]],
    )


def toffoli(N=None, controls=[0, 1], target=2):
    """Quantum object representing the Toffoli gate.

    Returns
    -------
    toff_gate : qobj
        Quantum object representation of Toffoli gate.

    Examples
    --------
    >>> toffoli() # doctest: +SKIP
    Quantum object: dims=[[2, 2, 2], [2, 2, 2]], shape = [8, 8], type='oper', dtype=Dense, isherm=True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  1.+0.j  0.+0.j]]


    """
    warnings.warn(
        "toffoli has been deprecated and will be removed in future version. \
        Use TOFFOLI.get_qobj() instead.", DeprecationWarning, stacklevel=2
    )
    if [controls[0], controls[1], target] != [0, 1, 2] and N is None:
        N = 3

    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(
            toffoli(), dims=[2] * N, targets=tuple(controls) + (target,)
        )
    return Qobj(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dims=[[2, 2, 2], [2, 2, 2]],
    )


#
# Miscellaneous Gates
#


def rotation(op, phi, N=None, target=0):
    """Single-qubit rotation for operator op with angle phi.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    if N is not None:
        _deprecation_warnings_gate_expansion()
        return expand_operator(rotation(op, phi), N, target)
    return (-1j * op * phi / 2).expm()


def globalphase(theta, N=1):
    """
    Returns quantum object representing the global phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of global phase shift gate.

    Examples
    --------
    >>> phasegate(pi/4) # doctest: +SKIP
    Quantum object: dims=[[2], [2]], \
shape = [2, 2], type='oper', dtype=Dense, isherm=False
    Qobj data =
    [[ 0.70710678+0.70710678j          0.00000000+0.j]
     [ 0.00000000+0.j          0.70710678+0.70710678j]]

    """
    warnings.warn(
        "global_phase has been deprecated and will be removed in future version. \
        Use GLOBALPHASE(phase).get_qobj() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    data = np.exp(1.0j * theta) * sp.eye(
        2**N, 2**N, dtype=complex, format="csr"
    )
    return Qobj(data, dims=[[2] * N, [2] * N])


#
# Operation on Gates
#


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


def _powers(op, N):
    """
    Generator that yields powers of an operator `op`,
    through to `N`.
    """
    acc = qeye(op.dims[0])
    yield acc

    for _ in range(N - 1):
        acc *= op
        yield acc


def qubit_clifford_group(N=None, target=0):
    """
    Generates the Clifford group on a single qubit,
    using the presentation of the group given by Ross and Selinger
    (http://www.mathstat.dal.ca/~selinger/newsynth/).

    Parameters
    ----------

    N : int or None
        Number of qubits on which each operator is to be defined
        (default: 1).
    target : int
        Index of the target qubit on which the single-qubit
        Clifford operators are to act.

    Yields
    ------

    op : Qobj
        Clifford operators, represented as Qobj instances.

    """

    # The Ross-Selinger presentation of the single-qubit Clifford
    # group expresses each element in the form C_{ijk} = E^i X^j S^k
    # for gates E, X and S, and for i in range(3), j in range(2) and
    # k in range(4).
    #
    # We start by defining these gates. E is defined in terms of H,
    # \omega and S, so we define \omega and H first.
    w = np.exp(1j * 2 * np.pi / 8)
    H = snot()

    X = sigmax()
    S = phasegate(np.pi / 2)
    E = H * (S**3) * w**3

    for op in map(
        partial(reduce, mul),
        product(_powers(E, 3), _powers(X, 2), _powers(S, 4)),
    ):
        # partial(reduce, mul) returns a function that takes products
        # of its argument, by analogy to sum. Note that by analogy,
        # sum can be written as partial(reduce, add).

        # product(...) yields the Cartesian product of its arguments.
        # Here, each element is a tuple (E**i, X**j, S**k) such that
        # partial(reduce, mul) acting on the tuple yields E**i * X**j * S**k.

        # Finally, we optionally expand the gate.
        if N is not None:
            yield expand_operator(op, N, target)
        else:
            yield op
