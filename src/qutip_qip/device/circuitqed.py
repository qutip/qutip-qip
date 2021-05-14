import numpy as np

from qutip import qeye, tensor, destroy, basis
from .modelprocessor import ModelProcessor
from ..transpiler import to_chain_structure
from ..compiler import SCQubitsCompiler
from ..noise import ZZCrossTalk


__all__ = ['SCQubits']


class SCQubits(ModelProcessor):
    """
    A chain of superconducting qubits with fixed frequency.
    Single-qubit control is realized by rotation around the X and Y axis
    while two-qubit gates are implemented with Cross Resonance gates.
    A 3-level system is used to simulate the superconducting qubit system,
    in order to simulation leakage.
    Various types of interaction can be realized on a superconducting
    system, as a demonstration and
    for simplicity, we only use a ZX Hamiltonian for
    the two-qubit interaction.
    For details see https://arxiv.org/abs/2005.12667 and
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.052308.

    Parameters
    ----------
    num_qubits: int
        Number of qubits
    t1, t2: float or list, optional
        Coherence time for all qubit or each qubit
    zz_crosstalk: bool, optional
        if zz cross talk is included.
    **params:
        Keyword argument for hardware parameters, in the unit of GHz.
        Each can should be given as list:

        - ``wq``: qubit bare frequency, default 5.15 and 5.09
          for each pair of superconducting qubits,
          e.g. ``[5.15, 5.09, 5.15, ...]``
        - ``wr``: resonator bare frequency, default ``[5.96]*num_qubits``
        - ``g``: The coupling strength between the resonator and the qubits,
          default ``[0.1]*(num_qubits - 1)``
        - ``alpha``: anharmonicity for each superconducting qubit,
          default ``[-0.3]*num_qubits``
        - ``omega_single``: control strength for single-qubit gate,
          default ``[0.01]*num_qubits``
        - ``omega_cr``: control strength for cross resonance gate,
          default ``[0.01]*num_qubits``

    Attributes
    ----------
    dims: list
        Dimension of the subsystem, e.g. ``[3,3,3]``.
    pulse_mode: "discrete" or "continuous"
        Given pulse is treated as continuous pulses or discrete step functions.
    native_gates: list of str
        The native gate sets
    """
    def __init__(
            self, num_qubits, t1=None, t2=None, zz_crosstalk=False, **params):
        super(SCQubits, self).__init__(
            num_qubits, t1=t1, t2=t2)
        self.num_qubits = num_qubits
        self.dims = [3] * num_qubits
        self.pulse_mode = "continuous"
        self.params = {
            "wq": np.array(
                (
                    (5.15, 5.09) * int(np.ceil(self.num_qubits / 2))
                )[: self.num_qubits]
            ),
            "wr": self.to_array(5.96, num_qubits - 1),
            "alpha": self.to_array(-0.3, num_qubits),
            "g": self.to_array(0.1, 2 * (num_qubits - 1)),
            "omega_single": self.to_array(0.01, num_qubits),
            "omega_cr": self.to_array(0.01, num_qubits)
        }
        if params is not None:
            self.params.update(params)
        self.set_up_ops()
        self.set_up_params()
        self.native_gates = ["RX", "RY", "CNOT"]
        self._default_compiler = SCQubitsCompiler
        if zz_crosstalk:
            self.add_noise(ZZCrossTalk(self.params))

    def set_up_ops(self):
        """
        Setup the operators.
        We use 2π σ/2 as the single-qubit control Hamiltonian and
        -2πZX/4 as the two-qubit Hamiltonian.
        """
        for m in range(self.num_qubits):
            destroy_op = destroy(self.dims[m])
            coeff = 2 * np.pi * self.params["alpha"][m] / 2.
            self.add_drift(
                coeff * destroy_op.dag() ** 2 * destroy_op ** 2, targets=[m])

        for m in range(self.num_qubits):
            destroy_op = destroy(self.dims[m])
            op = destroy_op + destroy_op.dag()
            self.add_control(2 * np.pi / 2 * op, [m], label="sx" + str(m))

        for m in range(self.num_qubits):
            destroy_op = destroy(self.dims[m])
            op = destroy_op * (-1.j) + destroy_op.dag() * 1.j
            self.add_control(2 * np.pi / 2 * op, [m], label="sy" + str(m))

        for m in range(self.num_qubits - 1):
            # For simplicity, we neglect leakage in two-qubit gates.
            d1 = self.dims[m]
            d2 = self.dims[m+1]
            # projector to the 0 and 1 subspace
            projector1 = basis(d1, 0) * basis(d1, 0).dag() + \
                basis(d1, 1) * basis(d2, 1).dag()
            projector2 = basis(d2, 0) * basis(d2, 0).dag() + \
                basis(d2, 1) * basis(d2, 1).dag()
            destroy_op1 = destroy(d1)
            # Notice that this is actually 2πZX/4
            z = projector1 * \
                (- destroy_op1.dag()*destroy_op1 * 2 + qeye(d1)) / 2  \
                * projector1
            destroy_op2 = destroy(d2)
            x = projector2 * (destroy_op2.dag() + destroy_op2) / 2 * projector2
            self.add_control(
                2 * np.pi * tensor([z, x]), [m, m+1],
                label="zx" + str(m) + str(m + 1)
            )
            self.add_control(
                2 * np.pi * tensor([x, z]), [m, m+1],
                label="zx" + str(m + 1) + str(m)
            )

    def set_up_params(self):
        """
        Compute the dressed frequency and the interaction strength.
        """
        g = self.params["g"]
        wq = self.params["wq"]
        wr = self.params["wr"]
        alpha = self.params["alpha"]
        # Dressed qubit frequency
        wq_dr = []
        for i in range(self.num_qubits):
            tmp = wq[i]
            if i != 0:
                tmp += g[2*i - 1]**2/(wq[i] - wr[i - 1])
            if i != (self.num_qubits - 1):
                tmp += g[2*i]**2/(wq[i] - wr[i])
            wq_dr.append(tmp)
        self.params["wq_dressed"] = wq_dr
        # Dressed resonator frequency
        wr_dr = []
        for i in range(self.num_qubits - 1):
            tmp = wr[i]
            tmp -= g[2*i]**2/(wq[i] - wr[i] + alpha[i])
            tmp -= g[2*i + 1]**2/(wq[i+1] - wr[i] + alpha[i])
            wr_dr.append(tmp)
        self.params["wr_dressed"] = wr_dr
        # Effective qubit coupling strength
        J = []
        for i in range(self.num_qubits - 1):
            tmp = g[2*i] * g[2*i+1] * \
                (wq[i] + wq[i+1] - 2*wr[i]) / \
                2 / (wq[i] - wr[i]) / (wq[i+1] - wr[i])
            J.append(tmp)
        self.params["J"] = J
        # Effective ZX strength
        zx_coeff = []
        omega_cr = self.params["omega_cr"]
        for i in range(self.num_qubits - 1):
            tmp = J[i] * omega_cr[i] * (
                1/(wq[i] - wq[i+1] + alpha[i]) -
                1/(wq[i] - wq[i+1])
                )
            zx_coeff.append(tmp)
        for i in range(self.num_qubits - 1, 0, -1):
            tmp = J[i-1] * omega_cr[i] * (
                1/(wq[i] - wq[i-1] + alpha[i]) -
                1/(wq[i] - wq[i-1])
                )
            zx_coeff.append(tmp)
        # Times 2 because we use -2πZX/4 as operators
        self.params["zx_coeff"] = np.asarray(zx_coeff) * 2

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        labels = [[r"$\sigma_x^%d$" % n for n in range(self.num_qubits)],
                [r"$\sigma_y^%d$" % n for n in range(self.num_qubits)]]
        label_zx = []
        for m in range(self.num_qubits - 1):
            label_zx.append(r"$ZX^{%d%d}$"% (m, m + 1))
            label_zx.append(r"$ZX^{%d%d}$"% (m + 1, m))
        labels.append(label_zx)
        return labels

    def topology_map(self, qc):
        return to_chain_structure(qc)
