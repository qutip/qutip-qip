import numpy as np

from qutip_qip.circuit import GateInstruction
from qutip_qip.operations import RX, RY, RZX
from qutip_qip.compiler import GateCompiler, PulseInstruction


class SCQubitsCompiler(GateCompiler):
    r"""
    Compiler for :class:`.SCQubits`.
    Compiled pulse strength is in the unit of GHz.

    Supported native gates: "RX", "RY", "CX".

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
    >>> from qutip_qip.operations import CX
    >>>
    >>> qc = QubitCircuit(2)
    >>> qc.add_gate(CX, targets=0, controls=1)
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
                "CX": self.cnot_compiler,
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

    def _rotation_compiler(
        self, circuit_instruction, op_label, param_label, args
    ):
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
        A list of :obj:`.PulseInstruction`, including the compiled pulse
        information for this gate.
        """
        target = circuit_instruction.targets[0]
        coeff, tlist = self.generate_pulse_shape(
            args["shape"],
            args["num_samples"],
            maximum=self.params[param_label][target],
            area=circuit_instruction.operation.arg_value[0] / 2.0 / np.pi,
        )
        f = 2 * np.pi * self.params["wq"][target]  # FIXME unused variable
        if args["DRAG"]:
            pulse_info = self._drag_pulse(op_label, coeff, tlist, target)
        elif op_label == "sx":
            pulse_info = [
                ("sx" + str(target), coeff),
                # Add zero here just to make it easier to add the driving frequency later.
                ("sy" + str(target), np.zeros(len(coeff))),
            ]
        elif op_label == "sy":
            pulse_info = [
                ("sx" + str(target), np.zeros(len(coeff))),
                ("sy" + str(target), coeff),
            ]
        else:
            raise RuntimeError("Unknown label.")
        return [PulseInstruction(circuit_instruction, tlist, pulse_info)]

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

    def ry_compiler(self, circuit_instruction, args):
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
        A list of :obj:`.PulseInstruction`, including the compiled pulse
        information for this gate.
        """
        return self._rotation_compiler(
            circuit_instruction, "sy", "omega_single", args
        )

    def rx_compiler(self, circuit_instruction, args):
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
        A list of :obj:`.PulseInstruction`, including the compiled pulse
        information for this gate.
        """
        return self._rotation_compiler(
            circuit_instruction, "sx", "omega_single", args
        )

    def rzx_compiler(self, circuit_instruction, args):
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
        A list of :obj:`.PulseInstruction`, including the compiled pulse
        information for this gate.
        """
        result = []
        q1, q2 = circuit_instruction.targets
        arg_value = circuit_instruction.operation.arg_value

        if q1 < q2:
            zx_coeff = self.params["zx_coeff"][2 * q1]
        else:
            zx_coeff = self.params["zx_coeff"][2 * q1 - 1]

        area = 0.5
        coeff, tlist = self.generate_pulse_shape(
            args["shape"], args["num_samples"], maximum=zx_coeff, area=area
        )
        area_rescale_factor = np.sqrt(np.abs(arg_value) / (np.pi / 2))
        tlist *= area_rescale_factor
        coeff *= area_rescale_factor
        pulse_info = [("zx" + str(q1) + str(q2), coeff)]
        result += [PulseInstruction(circuit_instruction, tlist, pulse_info)]
        return result

    def cnot_compiler(self, circuit_instruction, args):
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
        A list of :obj:`.PulseInstruction`, including the compiled pulse
        information for this gate.
        """
        PI = np.pi
        result = []
        q1 = circuit_instruction.controls[0]
        q2 = circuit_instruction.targets[0]

        # += extends a list in Python
        result += self.gate_compiler["RX"](
            GateInstruction(operation=RX(arg_value=-PI / 2), qubits=(q2,)),
            args,
        )

        result += self.gate_compiler["RZX"](
            GateInstruction(operation=RZX(arg_value=PI / 2), qubits=(q1, q2)),
            args,
        )

        result += self.gate_compiler["RX"](
            GateInstruction(operation=RX(arg_value=-PI / 2), qubits=(q1,)),
            args,
        )

        result += self.gate_compiler["RY"](
            GateInstruction(operation=RY(arg_value=-PI / 2), qubits=(q1,)),
            args,
        )

        result += self.gate_compiler["RX"](
            GateInstruction(operation=RX(arg_value=PI / 2), qubits=(q1,)),
            args,
        )

        return result
