import math
import os
import tempfile

import pytest
import numpy as np
from qutip import qeye
from qutip_qip.decompose.cnot_rotation import CNOT_ROTATION_BASIS
from qutip_qip.operations import BloqBuilder, decompose, expand_operator
from qutip_qip.operations.gates import CX, RX, RY, RZ


@pytest.fixture
def in_temporary_directory():
    """
    Creates a temporary directory for the lifetime of the fixture and changes
    into it.  All relative paths used will be in the temporary directory, and
    everything will automatically be cleaned up at the end of the fixture's
    life.
    """
    previous_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temporary_dir:
        os.chdir(temporary_dir)
        yield
        # pytest should catch exceptions occurring in functions using the
        # fixture, so this should always be called.  We want it here rather
        # than outside to prevent the case of the directory failing to be
        # removed because it is 'busy'.
        os.chdir(previous_dir)


# TODO: This is temporary, this should be implemented in QuantumCircuit
# need to remove this and its usage in the future.
def _gate_matrix(*ops):
    builder = BloqBuilder(ops[0].num_qreg)
    for op in ops:
        builder.add(op, builder.qreg)
    bloq = decompose(
        builder.to_bloq(), CNOT_ROTATION_BASIS, frozenset({CX, RX, RY, RZ})
    )

    dims = list(bloq.qreg_dim + bloq.aux_qreg_dim)
    U = qeye(dims)
    for ins in bloq.instructions:
        U = expand_operator(ins.op.get_qobj(), dims=dims, targets=ins.qreg) * U

    aux_dim = math.prod(bloq.aux_qreg_dim)
    return np.exp(1j * bloq.global_phase) * U.full()[::aux_dim, ::aux_dim]


@pytest.fixture
def gate_matrix():
    return _gate_matrix
