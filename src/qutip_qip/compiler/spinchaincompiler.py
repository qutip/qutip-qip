from functools import partial
import numpy as np

from ..circuit import QubitCircuit
from ..operations import Gate
from ..compiler import GateCompiler, Instruction


__all__ = ["SpinChainCompiler"]


class SpinChainCompiler(GateCompiler):
    """
    Compiler for :obj:`.SpinChain`.
    Compiled pulse strength is in the unit of MHz.

    Supported native gates: "RX", "RY", "RZ", "ISWAP", "SQRTISWAP",
    "GLOBALPHASE".

    Default configuration (see :obj:`.GateCompiler.args` and
    :obj:`.GateCompiler.compile`):

        +-----------------+--------------------+
        | key             | value              |
        +=================+====================+
        | ``shape``       | ``rectangular``    |
        +-----------------+--------------------+
        |``params``       | Hardware Parameters|
        +-----------------+--------------------+

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters.
        See :obj:`.SpinChainModel` for the definition.

    setup: string
        "linear" or "circular" for two sub-classes.

    global_phase: bool
        Record of the global phase change and will be returned.

    pulse_dict: dict, optional
        A map between the pulse label and its index in the pulse list.
        If given, the compiled pulse can be identified with
        ``(pulse_label, coeff)``, instead of ``(pulse_index, coeff)``.
        The number of key-value pairs should match the number of pulses
        in the processor.
        If it is empty, an integer ``pulse_index`` needs to be used
        in the compiling routine saved under the attributes ``gate_compiler``.

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

    setup: string
        "linear" or "circular" for two sub-classes.

    global_phase: bool
        Record of the global phase change and will be returned.

    Examples
    --------
    >>> import numpy as np
    >>> from qutip_qip.circuit import QubitCircuit
    >>> from qutip_qip.device import ModelProcessor, SpinChainModel
    >>> from qutip_qip.compiler import SpinChainCompiler
    >>>
    >>> qc = QubitCircuit(2)
    >>> qc.add_gate("RX", 0, arg_value=np.pi)
    >>> qc.add_gate("RZ", 1, arg_value=np.pi)
    >>>
    >>> model = SpinChainModel(2, "linear", g=0.1)
    >>> processor = ModelProcessor(model=model)
    >>> compiler = SpinChainCompiler(2, params=model.params, setup="linear")
    >>> processor.load_circuit(
    ...     qc, compiler=compiler) # doctest: +NORMALIZE_WHITESPACE
    ({'sx0': array([0., 1.]), 'sz1': array([0.  , 0.25])},
    {'sx0': array([0.25]), 'sz1': array([1.])})

    Notice that the above example is equivalent to using directly
    the :obj:`.LinearSpinChain`.
    """

    def __init__(
        self,
        num_qubits,
        params,
        setup="linear",
        global_phase=0.0,
        pulse_dict=None,
        N=None,
    ):
        super(SpinChainCompiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict, N=N
        )
        self.gate_compiler.update(
            {
                "ISWAP": self.iswap_compiler,
                "SQRTISWAP": self.sqrtiswap_compiler,
                "RZ": self.rz_compiler,
                "RX": self.rx_compiler,
                "GLOBALPHASE": self.globalphase_compiler,
            }
        )
        self.global_phase = global_phase

    def _rotation_compiler(self, gate, op_label, param_label, args):
        """
        Single qubit rotation compiler.

        Parameters
        ----------
        gate : :obj:`.Gate`:
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
            # The operator is Pauli Z/X/Y, without 1/2.
            area=gate.arg_value / 2.0 / np.pi * 0.5,
        )
        pulse_info = [(op_label + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def rz_compiler(self, gate, args):
        """
        Compiler for the RZ gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
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
        return self._rotation_compiler(gate, "sz", "sz", args)

    def rx_compiler(self, gate, args):
        """
        Compiler for the RX gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
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
        return self._rotation_compiler(gate, "sx", "sx", args)

    def _swap_compiler(self, gate, area, args):
        targets = gate.targets
        q1, q2 = min(targets), max(targets)
        g = self.params["sxsy"][q1]
        maximum = g
        coeff, tlist = self.generate_pulse_shape(
            args["shape"], args["num_samples"], maximum, area
        )
        if self.N != 2 and q1 == 0 and q2 == self.N - 1:
            pulse_name = "g" + str(q2)
        else:
            pulse_name = "g" + str(q1)
        pulse_info = [(pulse_name, coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def iswap_compiler(self, gate, args):
        """
        Compiler for the ISWAP gate.

        Parameters
        ----------
        gate : :obj:`.Gate`:
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
        return self._swap_compiler(gate, area=-1 / 8, args=args)

    def sqrtiswap_compiler(self, gate, args):
        """
        Compiler for the SQRTISWAP gate.

        Parameters
        ----------
        gate : :obj:`Gate`:
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
        return self._swap_compiler(gate, area=-1 / 16, args=args)

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
