import numbers
from packaging.version import parse as parse_version
from collections.abc import Iterable
from itertools import product
from functools import partial, reduce
from operator import mul
from typing import Optional, List, Union, Tuple, Generator

import warnings

import numpy as np
import scipy.sparse as sp

import qutip
from qutip import Qobj, identity, qeye, sigmax, sigmay, sigmaz, tensor

QubitSpecifier = Union[int, Iterable[int]]

__all__ = [
    "rx",
    "ry",
    "rz",
    "sqrtnot",
    "snot",
    "phasegate",
    "qrot",
    "x_gate",
    "y_gate",
    "z_gate",
    "cy_gate",
    "cz_gate",
    "s_gate",
    "t_gate",
    "qasmu_gate",
    "cs_gate",
    "ct_gate",
    "cphase",
    "cnot",
    "csign",
    "berkeley",
    "swapalpha",
    "swap",
    "iswap",
    "sqrtswap",
    "sqrtiswap",
    "fredkin",
    "molmer_sorensen",
    "toffoli",
    "rotation",
    "controlled_gate",
    "globalphase",
    "hadamard_transform",
    "qubit_clifford_group",
    "expand_operator",
    "gate_sequence_product",
]


#
# Single Qubit Gates
#


def x_gate() -> Qobj:
    """Pauli-X gate or sigmax operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the x-axis.

    """
    return sigmax()


def y_gate() -> Qobj:
    """Pauli-Y gate or sigmay operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the y-axis.

    """
    return sigmay()


def cy_gate() -> Qobj:
    """Controlled Y gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dims=[[2, 2], [2, 2]],
    )


def z_gate() -> Qobj:
    """Pauli-Z gate or sigmaz operator.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a single-qubit rotation through pi radians around the z-axis.

    """
    return sigmaz()


def cz_gate() -> Qobj:
    """Controlled Z gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dims=[[2, 2], [2, 2]],
    )


def s_gate() -> Qobj:
    """Single-qubit rotation also called Phase gate or the Z90 gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing
        a 90 degree rotation around the z-axis.

    """
    return Qobj([[1, 0], [0, 1j]])


def cs_gate() -> Qobj:
    """Controlled S gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]],
        dims=[[2, 2], [2, 2]],
    )


def t_gate() -> Qobj:
    """Single-qubit rotation related to the S gate by the relationship S=T*T.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing a phase shift of pi/4.

    """
    return Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def ct_gate() -> Qobj:
    """Controlled T gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * np.pi / 4)],
        ],
        dims=[[2, 2], [2, 2]],
    )


def rx(phi: float) -> Qobj:
    """Single-qubit rotation for operator sigmax with angle phi.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [
            [np.cos(phi / 2), -1j * np.sin(phi / 2)],
            [-1j * np.sin(phi / 2), np.cos(phi / 2)],
        ]
    )


def ry(phi: float) -> Qobj:
    """Single-qubit rotation for operator sigmay with angle phi.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [
            [np.cos(phi / 2), -np.sin(phi / 2)],
            [np.sin(phi / 2), np.cos(phi / 2)],
        ]
    )


def rz(phi: float) -> Qobj:
    """Single-qubit rotation for operator sigmaz with angle phi.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the rotation.

    """
    return Qobj([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])


def sqrtnot() -> Qobj:
    """Single-qubit square root NOT gate.

    Returns
    -------
    result : :class:`qutip.Qobj`
        Quantum object for operator describing the square root NOT gate.

    """
    return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]])


def snot() -> Qobj:
    """Quantum object representing the SNOT (Hadamard) gate.

    Returns
    -------
    snot_gate : :class:`qutip.Qobj`
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
    return 1 / np.sqrt(2.0) * Qobj([[1, 1], [1, -1]])


def phasegate(theta: float) -> Qobj:
    """
    Returns quantum object representing the phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : :class:`qutip.Qobj`
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
    return Qobj([[1, 0], [0, np.exp(1.0j * theta)]], dims=[[2], [2]])


def qrot(theta: float, phi: float) -> Qobj:
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


def qasmu_gate(args: Tuple[float, float, float]) -> Qobj:
    """
    QASM U-gate as defined in the OpenQASM standard.

    Parameters
    ----------
    args : tuple
        Three parameters:

        - theta : float
            The argument supplied to the last RZ rotation.
        - phi : float
            The argument supplied to the middle RY rotation.
        - gamma : float
            The argument supplied to the first RZ rotation.

    Returns
    -------
    qasmu_gate : :class:`qutip.Qobj`
        Quantum object representation of the QASM U-gate as defined in the
        OpenQASM standard.
    """

    theta, phi, gamma = args
    return Qobj(rz(phi) * ry(theta) * rz(gamma))


#
# 2 Qubit Gates
#


def cphase(theta: float) -> Qobj:
    """
    Returns quantum object representing the controlled phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    U : :class:`qutip.Qobj`
        Quantum object representation of controlled phase gate.
    """
    mat = np.identity(4, dtype=np.complex128)
    mat[2:, 2:] = phasegate(theta).full()
    return Qobj(mat, dims=[[2, 2], [2, 2]])


def cnot() -> Qobj:
    """
    Quantum object representing the CNOT gate.

    Returns
    -------
    cnot_gate : :class:`qutip.Qobj`
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
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dims=[[2, 2], [2, 2]],
    )


def csign() -> Qobj:
    """
    Quantum object representing the CSIGN gate.

    Returns
    -------
    csign_gate : :class:`qutip.Qobj`
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
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dims=[[2, 2], [2, 2]],
    )


def berkeley() -> Qobj:
    """
    Quantum object representing the Berkeley gate.

    Returns
    -------
    berkeley_gate : :class:`qutip.Qobj`
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
    return Qobj(
        [
            [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
            [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
            [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
            [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
        ],
        dims=[[2, 2], [2, 2]],
    )


def swapalpha(alpha: float) -> Qobj:
    """
    Quantum object representing the SWAPalpha gate.

    Returns
    -------
    swapalpha_gate : :class:`qutip.Qobj`
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


def swap() -> Qobj:
    """Quantum object representing the SWAP gate.

    Returns
    -------
    swap_gate : :class:`qutip.Qobj`
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
    return Qobj(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]],
    )


def iswap() -> Qobj:
    """Quantum object representing the iSWAP gate.

    Returns
    -------
    iswap_gate : :class:`qutip.Qobj`
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
    return Qobj(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]],
    )


def sqrtswap() -> Qobj:
    """Quantum object representing the square root SWAP gate.

    Returns
    -------
    sqrtswap_gate : :class:`qutip.Qobj`
        Quantum object representation of square root SWAP gate

    """
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


def sqrtiswap() -> Qobj:
    """Quantum object representing the square root iSWAP gate.

    Returns
    -------
    sqrtiswap_gate : :class:`qutip.Qobj`
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


def molmer_sorensen(theta: float, phi: float = 0.0) -> Qobj:
    """
    Quantum object of a Mølmer–Sørensen gate.

    Parameters
    ----------
    theta: float
        The duration of the interaction pulse.
    phi: float
        Rotation axis. phi = 0 for XX; phi=pi for YY

    Returns
    -------
    molmer_sorensen_gate : :class:`qutip.Qobj`
        Quantum object representation of the Mølmer–Sørensen gate.
    """
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


def fredkin() -> Qobj:
    """Quantum object representing the Fredkin gate.

    Returns
    -------
    fredkin_gate : :class:`qutip.Qobj`
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


def toffoli() -> Qobj:
    """Quantum object representing the Toffoli gate.

    Returns
    -------
    toff_gate : :class:`qutip.Qobj`
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


def rotation(op: Qobj, phi: float) -> Qobj:
    """Single-qubit rotation for operator op with angle phi.

    Returns
    -------
    result : ) -> Qobj:
        Quantum object for operator describing the rotation.

    """
    return (-1j * op * phi / 2).expm()


def controlled_gate(
    U: Qobj,
    controls: QubitSpecifier = 0,
    targets: QubitSpecifier = 1,
    N: Optional[int] = None,
    control_value: int = 1,
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
    result : :class:`qutip.Qobj`
        Quantum object representing the controlled-U gate.
    """
    # Compatibility
    if not isinstance(targets, Iterable):
        controls = [controls]
    if not isinstance(targets, Iterable):
        targets = [targets]
    num_controls = len(controls)
    num_targets = len(U.dims[0])
    N = num_controls + num_targets if N is None else N

    # First, assume that the last qubit is the target and control qubits are
    # in the increasing order.
    # The control_value is the location of this unitary.
    block_matrices = [np.array([[1, 0], [0, 1]])] * 2**num_controls
    block_matrices[control_value] = U.full()
    from scipy.linalg import block_diag  # move this to the top of the file

    result = block_diag(*block_matrices)
    result = Qobj(result, dims=[[2] * (num_controls + num_targets)] * 2)

    # Expand it to N qubits and permute qubits labelling
    if set(controls + targets) == set(range(N)):
        return result
    else:
        return expand_operator(result, N, targets=controls + targets)


def globalphase(theta: float, N: int = 1) -> Qobj:
    """
    Returns quantum object representing the global phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    Returns
    -------
    phase_gate : :class:`qutip.Qobj`
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
    data = np.exp(1.0j * theta) * sp.eye(
        2**N, 2**N, dtype=complex, format="csr"
    )
    return Qobj(data, dims=[[2] * N, [2] * N])


#
# Operation on Gates
#


def _hamming_distance(x: int, bits: int = 32) -> int:
    """
    Calculate the bit-wise Hamming distance of x from 0: That is, the number
    1s in the integer x.
    """
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot


def hadamard_transform(N: int = 1) -> Qobj:
    """Quantum object representing the N-qubit Hadamard gate.

    Returns
    -------
    q : :class:`qutip.Qobj`
        Quantum object representation of the N-qubit Hadamard gate.

    """
    data = [[1, 1], [1, -1]]
    H = Qobj(data) / np.sqrt(2)

    return tensor([H] * N)


def _powers(op: Qobj, N: int) -> Generator[Qobj, None, None]:
    """
    Generator that yields powers of an operator `op`,
    through to `N`.
    """
    acc = qeye(op.dims[0])
    yield acc

    for _ in range(N - 1):
        acc *= op
        yield acc


def qubit_clifford_group() -> Generator[Qobj, None, None]:
    """
    Generates the Clifford group on a single qubit,
    using the presentation of the group given by Ross and Selinger
    (http://www.mathstat.dal.ca/~selinger/newsynth/).

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
        yield op


#
# Gate Expand
#


def _check_oper_dims(
    oper: Qobj,
    dims: Optional[List[int]] = None,
    targets: Optional[QubitSpecifier] = None,
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
    # if operator matches N
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
                "The operator dims {} do not match "
                "the target dims {}.".format(oper.dims[0], targ_dims)
            )


def _targets_to_list(
    targets: Optional[QubitSpecifier],
    oper: Optional[Qobj] = None,
    N: Optional[int] = None,
) -> List[int]:
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
                "The given operator needs {} "
                "target qutbis, "
                "but {} given.".format(req_num, len(targets))
            )
    # if targets is smaller than N
    if N is not None:
        if not all([t < N for t in targets]):
            raise ValueError("Targets must be smaller than N={}.".format(N))
    return targets


def expand_operator(
    oper: Qobj,
    N: Optional[int] = None,
    targets: Optional[QubitSpecifier] = None,
    dims: Optional[List[int]] = None,
    cyclic_permutation: bool = False,
    dtype: Optional[str] = None,
) -> Union[Qobj, List[Qobj]]:
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
    >>> from qutip_qip.operations import expand_operator, x_gate, cnot
    >>> import qutip
    >>> expand_operator(x_gate(), dims=[2,3], targets=[0]) # doctest: +NORMALIZE_WHITESPACE
    Quantum object: dims=[[2, 3], [2, 3]], shape=(6, 6), type='oper', dtype=CSR, isherm=True
    Qobj data =
    [[0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]]
    >>> expand_operator(cnot(), dims=[2,2,2], targets=[1, 2]) # doctest: +NORMALIZE_WHITESPACE
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
    >>> expand_operator(cnot(), dims=[2, 2, 2], targets=[2, 0]) # doctest: +NORMALIZE_WHITESPACE
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
    if parse_version(qutip.__version__) >= parse_version("5.dev"):
        # If no data type specified, use CSR
        dtype = dtype or qutip.settings.core["default_dtype"] or qutip.data.CSR
        oper = oper.to(dtype)
    if N is not None:
        warnings.warn(
            "The function expand_operator has been generalized to "
            "arbitrary subsystems instead of only qubit systems."
            "Please use the new signature e.g.\n"
            "expand_operator(oper, dims=[2, 3, 2, 2], targets=2)",
            DeprecationWarning,
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
    if parse_version(qutip.__version__) >= parse_version("5.dev"):
        out = out.to(dtype)
    return out


def gate_sequence_product(
    U_list: List[Qobj],
    left_to_right: bool = True,
    inds_list: Optional[List[List[int]]] = None,
    expand: bool = False,
) -> Union[Qobj, Tuple[Qobj, List[int]]]:
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
    U_overall : :class:`qutip.Qobj`
        Unitary matrix corresponding to U_list.

    overall_inds : list of int, optional
        List of qubit indices on which U_overall applies.
    """
    from ..circuit.circuitsimulator import (
        _gate_sequence_product,
        _gate_sequence_product_with_expansion,
    )

    if expand:
        return _gate_sequence_product(U_list, inds_list)
    else:
        return _gate_sequence_product_with_expansion(U_list, left_to_right)
