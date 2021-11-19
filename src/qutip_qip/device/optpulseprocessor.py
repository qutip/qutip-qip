from collections.abc import Iterable
import warnings
import numbers

import numpy as np

from qutip import Qobj, identity, tensor, mesolve
import qutip.control.pulseoptim as cpo
from ..circuit import QubitCircuit
from .processor import Processor
from ..operations import gate_sequence_product, expand_operator


__all__ = ["OptPulseProcessor"]


class OptPulseProcessor(Processor):
    """
    A processor that uses
    :obj:`qutip.control.optimize_pulse_unitary`
    to find optimized pulses for a given quantum circuit.
    The processor can simulate the evolution under the given
    control pulses using :func:`qutip.mesolve`.
    (For attributes documentation, please
    refer to the parent class :class:`.Processor`)

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    drift: `:class:`qutip.Qobj`
        The drift Hamiltonian. The size must match the whole quantum system.

    dims: list
        The dimension of each component system.
        Default value is a qubit system of ``dim=[2,2,2,...,2]``

    **params:
        - t1 : float or list, optional
            Characterize the amplitude damping for each qubit.
            A list of size `num_qubits` or a float for all qubits.
        - t2 : float or list, optional
            Characterize the total dephasing for each qubit.
            A list of size `num_qubits` or a float for all qubits.
    """

    def __init__(self, num_qubits=None, drift=None, dims=None, **params):
        super(OptPulseProcessor, self).__init__(
            num_qubits, dims=dims, **params
        )
        if drift is not None:
            self.add_drift(drift, list(range(self.num_qubits)))
        self.spline_kind = "step_func"

    def load_circuit(
        self,
        qc,
        min_fid_err=np.inf,
        merge_gates=True,
        setting_args=None,
        verbose=False,
        **kwargs
    ):
        """
        Find the pulses realizing a given :class:`.Circuit` using
        :func:`qutip.control.optimize_pulse_unitary`. Further parameter for
        for :func:`qutip.control.optimize_pulse_unitary` needs to be given as
        keyword arguments. By default, it first merge all the gates
        into one unitary and then find the control pulses for it.
        It can be turned off and one can set different parameters
        for different gates. See examples for details.

        Examples
        --------
        Same parameter for all the gates

        >>> from qutip_qip.circuit import QubitCircuit
        >>> from qutip_qip.device import OptPulseProcessor
        >>> qc = QubitCircuit(1)
        >>> qc.add_gate("SNOT", 0)
        >>> num_tslots = 10
        >>> evo_time = 10
        >>> processor = OptPulseProcessor(1, drift=sigmaz())
        >>> processor.add_control(sigmax())
        >>> # num_tslots and evo_time are two keyword arguments
        >>> tlist, coeffs = processor.load_circuit(\
                qc, num_tslots=num_tslots, evo_time=evo_time)

        Different parameters for different gates

        >>> from qutip_qip.circuit import QubitCircuit
        >>> from qutip_qip.device import OptPulseProcessor
        >>> qc = QubitCircuit(2)
        >>> qc.add_gate("SNOT", 0)
        >>> qc.add_gate("SWAP", targets=[0, 1])
        >>> qc.add_gate('CNOT', controls=1, targets=[0])
        >>> processor = OptPulseProcessor(2, drift=tensor([sigmaz()]*2))
        >>> processor.add_control(sigmax(), cyclic_permutation=True)
        >>> processor.add_control(sigmay(), cyclic_permutation=True)
        >>> processor.add_control(tensor([sigmay(), sigmay()]))
        >>> setting_args = {"SNOT": {"num_tslots": 10, "evo_time": 1},\
                        "SWAP": {"num_tslots": 30, "evo_time": 3},\
                        "CNOT": {"num_tslots": 30, "evo_time": 3}}
        >>> tlist, coeffs = processor.load_circuit(\
                qc, setting_args=setting_args, merge_gates=False)

        Parameters
        ----------
        qc : :class:`.QubitCircuit` or list of Qobj
            The quantum circuit to be translated.

        min_fid_err: float, optional
            The minimal fidelity tolerance, if the fidelity error of any
            gate decomposition is higher, a warning will be given.
            Default is infinite.

        merge_gates: boolean, optimal
            If True, merge all gate/Qobj into one Qobj and then
            find the optimal pulses for this unitary matrix. If False,
            find the optimal pulses for each gate/Qobj.

        setting_args: dict, optional
            Only considered if merge_gates is False.
            It is a dictionary containing keyword arguments
            for different gates.

        verbose: boolean, optional
            If true, the information for each decomposed gate
            will be shown. Default is False.

        **kwargs
            keyword arguments for
            :func:``qutip.control.optimize_pulse_unitary``

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape ``(len(ctrls), len(tlist)-1)``. Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.

        Notes
        -----
        ``len(tlist)-1=coeffs.shape[1]`` since tlist gives
        the beginning and the end of the pulses
        """
        if setting_args is None:
            setting_args = {}
        if isinstance(qc, QubitCircuit):
            props = qc.propagators()
            gates = [g.name for g in qc.gates]
        elif isinstance(qc, Iterable):
            props = qc
            gates = None  # using list of Qobj, no gates name
        else:
            raise ValueError(
                "qc should be a " "QubitCircuit or a list of Qobj"
            )
        if merge_gates:  # merge all gates/Qobj into one Qobj
            props = [gate_sequence_product(props)]
            gates = None

        time_record = []  # a list for all the gates
        coeff_record = []
        last_time = 0.0  # used in concatenation of tlist
        for prop_ind, U_targ in enumerate(props):
            U_0 = identity(U_targ.dims[0])

            # If qc is a QubitCircuit and setting_args is not empty,
            # we update the kwargs for each gate.
            # keyword arguments in setting_arg have priority
            if gates is not None and setting_args:
                kwargs.update(setting_args[gates[prop_ind]])

            control_labels = self.model.get_control_labels()
            full_ctrls_hams = []
            for label in control_labels:
                qobj, targets = self.model.get_control(label)
                full_ctrls_hams.append(
                    expand_operator(
                        qobj, len(self.dims), targets=targets, dims=self.dims
                    )
                )

            full_drift_ham = sum(
                [
                    expand_operator(
                        qobj, len(self.dims), targets=targets, dims=self.dims
                    )
                    for (qobj, targets) in self.model.get_all_drift()
                ],
                Qobj(
                    np.zeros(full_ctrls_hams[0].shape),
                    dims=[self.dims, self.dims],
                ),
            )

            result = cpo.optimize_pulse_unitary(
                full_drift_ham, full_ctrls_hams, U_0, U_targ, **kwargs
            )

            if result.fid_err > min_fid_err:
                warnings.warn(
                    "The fidelity error of gate {} is higher "
                    "than required limit. Use verbose=True to see"
                    "the more detailed information.".format(prop_ind)
                )

            time_record.append(result.time[1:] + last_time)
            last_time += result.time[-1]
            coeff_record.append(result.final_amps.T)

            if verbose:
                print("********** Gate {} **********".format(prop_ind))
                print("Final fidelity error {}".format(result.fid_err))
                print(
                    "Final gradient normal {}".format(result.grad_norm_final)
                )
                print("Terminated due to {}".format(result.termination_reason))
                print("Number of iterations {}".format(result.num_iter))

        tlist = np.hstack([[0.0]] + time_record)
        for i in range(len(self.pulses)):
            self.pulses[i].tlist = tlist
        coeffs = np.vstack([np.hstack(coeff_record)])

        coeffs = {label: coeff for label, coeff in zip(control_labels, coeffs)}
        self.set_coeffs(coeffs)
        self.set_tlist(tlist)
        return tlist, coeffs
