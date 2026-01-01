from qutip import basis, destroy, qeye, tensor
from qutip_qip.noise import Noise
from qutip_qip.pulse import Pulse


class ZZCrossTalk(Noise):
    """
    An always-on ZZ cross talk noise with the corresponding coefficient
    on each pair of qubits.
    The operator acts only one the lowerest two levels and
    is 0 on higher level.
    Equivalent to ``tensor(sigmaz(), sigmaz())``.

    Parameters
    ----------
    params:
        Parameters computed from a :class:`.SCQubits`.
    """

    def __init__(self, params):
        self.params = params

    def get_noisy_pulses(self,
        dims: list[int] | None = None,
        pulses: list[Pulse] | None = None,
        systematic_noise: Pulse | None = None
    ) -> tuple[list[Pulse], Pulse]:
        """
        Return the input pulses list with noise added and
        the pulse independent noise in a dummy :class:`.Pulse` object.

        Parameters
        ----------
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.
        pulses : list of :class:`.Pulse`
            The input pulses. The noise will be added to pulses in this list.
        systematic_noise : :class:`.Pulse`
            The dummy pulse with no ideal control element.

        Returns
        -------
        noisy_pulses: list of :class:`.Pulse`
            Noisy pulses.
        systematic_noise : :class:`.Pulse`
            The dummy pulse representing pulse-independent noise.
        """

        #TODO check these unused parameters
        J = self.params["J"]
        wr_dr = self.params["wr_dressed"]
        wr = self.params["wr"]
        wq_dr_cav = self.params["wq_dressed_cavity"]
        wq_dr = self.params["wq_dressed"]
        wq = self.params["wq"]
        alpha = self.params["alpha"]
        omega = self.params["omega_cr"]

        for i in range(len(dims) - 1):
            d1 = dims[i]
            d2 = dims[i + 1]
            destroy_op1 = destroy(d1)
            destroy_op2 = destroy(d2)

            projector1 = (
                basis(d1, 0) * basis(d1, 0).dag()
                + basis(d1, 1) * basis(d2, 1).dag()
            )

            projector2 = (
                basis(d2, 0) * basis(d2, 0).dag()
                + basis(d2, 1) * basis(d2, 1).dag()
            )

            z1 = (
                projector1
                * (destroy_op1.dag() * destroy_op1 * 2 - qeye(d1))
                * projector1
            )

            z2 = (
                projector2
                * (destroy_op2.dag() * destroy_op2 * 2 - qeye(d1))
                * projector2
            )

            zz_op = tensor(z1, z2)
            zz_coeff = (
                1 / (wq_dr_cav[i] - wq_dr_cav[i + 1] - alpha[i + 1])
                - 1 / (wq_dr_cav[i] - wq_dr_cav[i + 1] + alpha[i])
            ) * J[i] ** 2

            systematic_noise.add_control_noise(
                zz_coeff * zz_op / 2,
                targets=[i, i + 1],
                coeff=True,
                tlist=None,
            )

        return pulses, systematic_noise
