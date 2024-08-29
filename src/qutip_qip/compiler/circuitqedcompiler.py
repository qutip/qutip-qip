import numpy as np

from ..operations import Gate
from ..compiler import GateCompiler, Instruction


__all__ = ["SCQubitsCompiler"]


class SCQubitsCompiler(GateCompiler):
    r"""
    Compiler for :class:`.SCQubits`.
    Compiled pulse strength is in the unit of GHz.

    Supported native gates: "RX", "RY", "CNOT".

    Default configuration (see :obj:`.GateCompiler.args` and
    :obj:`.GateCompiler.compile`):

        +-----------------+-----------------------+
        | key             | value                 |
        +=================+=======================+
        | ``shape``       | ``hann``              |
        +-----------------+-----------------------+
        |``num_samples``  | 1000                  |
        +-----------------+-----------------------+
        |``params``       | Hardware Parameters   |
        +-----------------+-----------------------+

    For single-qubit gate, we apply the DRAG correction :cite:`motzoi2013`

    .. math::

        \Omega^{x} &= \Omega_0 - \frac{\Omega_0^3}{4 \alpha^2}

        \Omega^{y} &= - \frac{\dot{\Omega}_0 }{\alpha}

        \Omega^{z} &= - \frac{\Omega_0^2}{\alpha} + \frac{2 \Omega_0^2}{
            4 \alpha
        }

    where :math:`\Omega_0` is the original shape of the pulse.
    Notice that the :math:`\Omega_0` and its first derivative
    should be 0 from the starts and the end.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters.
        See :meth:`.SCQubitsModel` for the definition.

    Attributes
    ----------
    num_qubits: int
        The number of the component systems.

    params: dict
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.

    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.

    Examples
    --------
    >>> import numpy as np
    >>> from qutip_qip.circuit import QubitCircuit
    >>> from qutip_qip.device import ModelProcessor, SCQubitsModel
    >>> from qutip_qip.compiler import SCQubitsCompiler
    >>>
    >>> qc = QubitCircuit(2)
    >>> qc.add_gate("CNOT", targets=0, controls=1)
    >>>
    >>> model = SCQubitsModel(2)
    >>> processor = ModelProcessor(model=model)
    >>> compiler = SCQubitsCompiler(2, params=model.params)
    >>> processor.load_circuit(qc, compiler=compiler);  # doctest: +SKIP

    Notice that the above example is equivalent to using directly
    the :obj:`.SCQubits`.
    """

    def __init__(self, num_qubits, params):
        super(SCQubitsCompiler, self).__init__(num_qubits, params=params)
        self.gate_compiler.update(
            {
                "RY": self.ry_compiler,
                "RX": self.rx_compiler,
                "CNOT": self.cnot_compiler,
                "RZX": self.rzx_compiler,
            }
        )
        self.args = {  # Default configuration
            "shape": "hann",
            "num_samples": 101,
            "params": self.params,
            "DRAG": True,
        }

    def _rotation_compiler(self, gate, op_label, param_label, args):
        """
        Single qubit rotation compiler.

        Parameters
        ----------
        gate : :obj:`~.operations.Gate`:
            The quantum gate to be compiled.
        op_label : str
            Label of the corresponding control Hamiltonian.
        param_label : str
            Label of the hardware parameters saved in
            :obj:`GateCompiler.params`.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        targets = gate.targets
        coeff, tlist = self.generate_pulse_shape(
            args["shape"],
            args["num_samples"],
            maximum=self.params[param_label][targets[0]],
            area=gate.arg_value / 2.0 / np.pi,
        )
        f = 2 * np.pi * self.params["wq"][targets[0]]
        if args["DRAG"]:
            pulse_info = self._drag_pulse(op_label, coeff, tlist, targets[0])
        elif op_label == "sx":
            pulse_info = [
                ("sx" + str(targets[0]), coeff),
                # Add zero here just to make it easier to add the driving frequency later.
                ("sy" + str(targets[0]), np.zeros(len(coeff))),
            ]
        elif op_label == "sy":
            pulse_info = [
                ("sx" + str(targets[0]), np.zeros(len(coeff))),
                ("sy" + str(targets[0]), coeff),
            ]
        else:
            raise RuntimeError("Unknown label.")
        return [Instruction(gate, tlist, pulse_info)]

    def _drag_pulse(self, op_label, coeff, tlist, target):
        dt_coeff = np.gradient(coeff, tlist[1] - tlist[0]) / 2 / np.pi
        # Y-DRAG
        alpha = self.params["alpha"][target]
        y_drag = -dt_coeff / alpha
        # Z-DRAG
        z_drag = -(coeff**2) / alpha + (np.sqrt(2) ** 2 * coeff**2) / (
            4 * alpha
        )
        # X-DRAG
        coeff += -(coeff**3 / (4 * alpha**2))

        pulse_info = [
            (op_label + str(target), coeff),
            ("sz" + str(target), z_drag),
        ]
        if op_label == "sx":
            pulse_info.append(("sy" + str(target), y_drag))
        elif op_label == "sy":
            pulse_info.append(("sx" + str(target), -y_drag))
        return pulse_info

    def ry_compiler(self, gate, args):
        """
        Compiler for the RZ gate

        Parameters
        ----------
        gate : :obj:`~.operations.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        return self._rotation_compiler(gate, "sy", "omega_single", args)

    def rx_compiler(self, gate, args):
        """
        Compiler for the RX gate

        Parameters
        ----------
        gate : :obj:`~.operations.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        return self._rotation_compiler(gate, "sx", "omega_single", args)

    def rzx_compiler(self, gate, args):
        """
        Cross-Resonance RZX rotation, building block for the CNOT gate.

        Parameters
        ----------
        gate : :obj:`~.operations.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        result = []
        q1, q2 = gate.targets
        if q1 < q2:
            zx_coeff = self.params["zx_coeff"][2 * q1]
        else:
            zx_coeff = self.params["zx_coeff"][2 * q1 - 1]
        area = 0.5
        coeff, tlist = self.generate_pulse_shape(
            args["shape"], args["num_samples"], maximum=zx_coeff, area=area
        )
        area_rescale_factor = np.sqrt(np.abs(gate.arg_value) / (np.pi / 2))
        tlist *= area_rescale_factor
        coeff *= area_rescale_factor
        pulse_info = [("zx" + str(q1) + str(q2), coeff)]
        result += [Instruction(gate, tlist, pulse_info)]
        return result

    def cnot_compiler(self, gate, args):
        """
        Compiler for CNOT gate using the cross resonance iteraction.
        See
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.134507
        for reference.

        Parameters
        ----------
        gate : :obj:`~.operations.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        result = []
        q1 = gate.controls[0]
        q2 = gate.targets[0]

        gate1 = Gate("RX", q2, arg_value=-np.pi / 2)
        result += self.gate_compiler[gate1.name](gate1, args)

        gate2 = Gate("RZX", targets=[q1, q2], arg_value=np.pi / 2)
        result += self.rzx_compiler(gate2, args)

        gate3 = Gate("RX", q1, arg_value=-np.pi / 2)
        result += self.gate_compiler[gate3.name](gate3, args)
        gate4 = Gate("RY", q1, arg_value=-np.pi / 2)
        result += self.gate_compiler[gate4.name](gate4, args)
        gate5 = Gate("RX", q1, arg_value=np.pi / 2)
        result += self.gate_compiler[gate5.name](gate5, args)
        return result
