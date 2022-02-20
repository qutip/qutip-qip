import warnings
import numpy as np
from scipy import signal

from .instruction import Instruction
from .scheduler import Scheduler
from ..circuit import QubitCircuit
from ..operations import Gate


__all__ = ["GateCompiler"]


class GateCompiler(object):
    """
    Base class of compilers, including the :meth:`GateCompiler.compile` method.
    It compiles a :class:`.QubitCircuit` into
    the pulse sequence for the processor. The core member function
    `compile` calls compiling method from the sub-class and concatenate
    the compiled pulses.

    Parameters
    ----------
    num_qubits: int
        The number of the component systems.

    params: dict, optional
        A Python dictionary contains the name and the value of the parameters,
        such as laser frequency, detuning etc.
        It will be saved in the class attributes and can be used to calculate
        the control pulses.

    Attributes
    ----------
    gate_compiler: dict
        The Python dictionary in the form of {gate_name: compiler_function}.
        It saves the compiling routine for each gate. See sub-classes
        for implementation.
        Note that for continuous pulse, the first coeff should always be 0.

    args: dict
        The compilation configurations.
        It will be passed to each compiling functions.
        Available arguments:

        * ``shape``: The compiled pulse shape. ``rectangular`` or
          one of the `SciPy window functions
          <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_.
        * ``num_samples``:
          Number of samples for continuous pulses.
          It has no effect for rectangular pulses.
        * ``params``: Hardware parameters computed in the :obj:`Processor`.

    """

    def __init__(self, num_qubits=None, params=None, pulse_dict=None, N=None):
        self.gate_compiler = {}
        self.num_qubits = num_qubits or N
        self.N = num_qubits  # backward compatibility
        self.params = params if params is not None else {}
        self.gate_compiler = {
            "GLOBALPHASE": self.globalphase_compiler,
            "IDLE": self.idle_compiler,
        }
        self.args = {  # Default configuration
            "shape": "rectangular",
            "num_samples": None,
            "params": self.params,
        }
        self.global_phase = 0.0
        if pulse_dict is not None:
            warnings.warn(
                """
                Giving pulse_dict to compiler is deprecated.
                The compiler now returns the compiled pulses as a dictionary
                between the pulse's label and the coefficients/tlist.
                It can be given to the processor directly.
                The parameter pulse_dict has no effect now,
                you can simply remove it.
                """,
                DeprecationWarning,
            )

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        pass

    def idle_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        idle_time = gate.arg_value
        return [Instruction(gate, idle_time, [])]

    def compile(self, circuit, schedule_mode=None, args=None):
        """
        Compile the the native gates into control pulse sequence.
        It calls each compiling method and concatenates
        the compiled pulses.

        Parameters
        ----------
        circuit: :class:`.QubitCircuit` or list of
            :class:`.Gate`
            A list of elementary gates that can be implemented in the
            corresponding hardware.
            The gate names have to be in `gate_compiler`.

        schedule_mode: str, optional
            ``"ASAP"`` for "as soon as possible" or
            ``"ALAP"`` for "as late as possible" or
            ``False`` or ``None`` for no schedule.
            Default is None.

        args: dict, optional
            A dictionary of arguments used in a specific gate compiler
            function.

        Returns
        -------
        tlist, coeffs: array_like or dict
            Compiled ime sequence and pulse coefficients.
            if ``return_array`` is true, return
            A 2d NumPy array of the shape ``(len(ctrls), len(tlist))``.
            Each row corresponds to the control pulse sequence for
            one Hamiltonian.
            if ``return_array`` is false
        """
        if isinstance(circuit, QubitCircuit):
            gates = circuit.gates
        else:
            gates = circuit
        if args is not None:
            self.args.update(args)
        instruction_list = []

        # compile gates
        for gate in gates:
            if gate.name not in self.gate_compiler:
                raise ValueError("Unsupported gate %s" % gate.name)
            instruction = self.gate_compiler[gate.name](gate, self.args)
            if instruction is None:
                continue  # neglecting global phase gate
            instruction_list += instruction
        if not instruction_list:
            return None, None

        # schedule
        # scheduled_start_time:
        #   An ordered list of the start_time for each pulse,
        #   corresponding to gates in the instruction_list.
        # instruction_list reordered according to the scheduled result
        instruction_list, scheduled_start_time = self._schedule(
            instruction_list, schedule_mode
        )

        # An instruction can be composed from several different pulse elements.
        # We separate them an assign them to each pulse index.
        pulse_ind_map = {}
        next_pulse_ind = 0
        pulse_instructions = []
        for instruction, start_time in zip(
            instruction_list, scheduled_start_time
        ):
            for pulse_name, coeff in instruction.pulse_info:
                if pulse_name not in pulse_ind_map:
                    pulse_instructions.append([])
                    pulse_ind_map[pulse_name] = next_pulse_ind
                    next_pulse_ind += 1
                pulse_instructions[pulse_ind_map[pulse_name]].append(
                    (start_time, instruction.tlist, coeff)
                )

        # concatenate pulses
        compiled_tlist, compiled_coeffs = self._concatenate_pulses(
            pulse_instructions, scheduled_start_time, len(pulse_instructions)
        )
        compiled_tlist_map, compiled_coeffs_map = {}, {}
        for key, index in pulse_ind_map.items():
            compiled_tlist_map[key] = compiled_tlist[index]
            compiled_coeffs_map[key] = compiled_coeffs[index]
        return compiled_tlist_map, compiled_coeffs_map

    def _schedule(self, instruction_list, schedule_mode):
        """
        Schedule the instructions if required and
        reorder instruction_list accordingly
        """
        if schedule_mode:
            scheduler = Scheduler(schedule_mode)
            scheduled_start_time = scheduler.schedule(instruction_list)
            time_ordered_pos = np.argsort(scheduled_start_time)
            instruction_list = [instruction_list[i] for i in time_ordered_pos]
            scheduled_start_time.sort()
        else:  # no scheduling
            scheduled_start_time = [0.0]
            for instruction in instruction_list[:-1]:
                scheduled_start_time.append(
                    instruction.duration + scheduled_start_time[-1]
                )
        return instruction_list, scheduled_start_time

    def _concatenate_pulses(
        self, pulse_instructions, scheduled_start_time, num_controls
    ):
        """
        Concatenate compiled pulses coefficients and tlist for each pulse.
        If there is idling time, add zeros properly to prevent wrong spline.
        """
        # Concatenate tlist and coeffs for each control pulses
        compiled_tlist = [[] for tmp in range(num_controls)]
        compiled_coeffs = [[] for tmp in range(num_controls)]
        for pulse_ind in range(num_controls):
            last_pulse_time = 0.0
            for start_time, tlist, coeff in pulse_instructions[pulse_ind]:
                # compute the gate time, step size and coeffs
                # according to different pulse mode
                (
                    gate_tlist,
                    coeffs,
                    step_size,
                    pulse_mode,
                ) = self._process_gate_pulse(start_time, tlist, coeff)

                if abs(last_pulse_time) < step_size * 1.0e-6:  # if first pulse
                    compiled_tlist[pulse_ind].append([0.0])
                    if pulse_mode == "continuous":
                        compiled_coeffs[pulse_ind].append([0.0])
                    # for discrete pulse len(coeffs) = len(tlist) - 1

                # If there is idling time between the last pulse and
                # the current one, we need to add zeros in between.
                if np.abs(start_time - last_pulse_time) > step_size * 1.0e-6:
                    idling_tlist = self._process_idling_tlist(
                        pulse_mode, start_time, last_pulse_time, step_size
                    )
                    compiled_tlist[pulse_ind].append(idling_tlist)
                    compiled_coeffs[pulse_ind].append(
                        np.zeros(len(idling_tlist))
                    )

                # Add the gate time and coeffs to the list.
                execution_time = gate_tlist + start_time
                last_pulse_time = execution_time[-1]
                compiled_tlist[pulse_ind].append(execution_time)
                compiled_coeffs[pulse_ind].append(coeffs)

        for i in range(num_controls):
            if not compiled_coeffs[i]:
                compiled_tlist[i] = None
                compiled_coeffs[i] = None
            else:
                compiled_tlist[i] = np.concatenate(compiled_tlist[i])
                compiled_coeffs[i] = np.concatenate(compiled_coeffs[i])
        return compiled_tlist, compiled_coeffs

    def _process_gate_pulse(self, start_time, tlist, coeff):
        # compute the gate time, step size and coeffs
        # according to different pulse mode
        if np.isscalar(tlist):
            pulse_mode = "discrete"
            # a single constant rectanglar pulse, where
            # tlist and coeff are just float numbers
            step_size = tlist
            coeff = np.array([coeff])
            gate_tlist = np.array([tlist])
        elif len(tlist) - 1 == len(coeff):
            # discrete pulse
            pulse_mode = "discrete"
            step_size = tlist[1] - tlist[0]
            coeff = np.asarray(coeff)
            gate_tlist = np.asarray(tlist)[1:]  # first t always 0 by def
        elif len(tlist) == len(coeff):
            # continuos pulse
            pulse_mode = "continuous"
            step_size = tlist[1] - tlist[0]
            coeff = np.asarray(coeff)[1:]
            gate_tlist = np.asarray(tlist)[1:]
        else:
            raise ValueError("The shape of the compiled pulse is not correct.")
        return gate_tlist, coeff, step_size, pulse_mode

    def _process_idling_tlist(
        self, pulse_mode, start_time, last_pulse_time, step_size
    ):
        idling_tlist = []
        if pulse_mode == "continuous":
            # We add sufficient number of zeros at the begining
            # and the end of the idling to prevent wrong cubic spline.
            if start_time - last_pulse_time > 3 * step_size:
                idling_tlist1 = np.linspace(
                    last_pulse_time + step_size / 5,
                    last_pulse_time + step_size,
                    5,
                )
                idling_tlist2 = np.linspace(
                    start_time - step_size, start_time, 5
                )
                idling_tlist.extend([idling_tlist1, idling_tlist2])
            else:
                idling_tlist.append(
                    np.arange(
                        last_pulse_time + step_size, start_time, step_size
                    )
                )
        elif pulse_mode == "discrete":
            # idling until the start time
            idling_tlist.append([start_time])
        return np.concatenate(idling_tlist)

    @classmethod
    def generate_pulse_shape(cls, shape, num_samples, maximum=1.0, area=1.0):
        """
        Return a tuple consisting of a coeff list and a time sequence
        according to a given pulse shape.

        Parameters
        ----------
        shape : str
            The name ``"rectangular"`` for constant pulse or
            the name of a Scipy window function.
            See
            `the Scipy documentation
            <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_
            for detail.
        num_samples : int
            The number of the samples of the coefficients.
        maximum : float, optional
            The maximum of the coefficients.
            The absolute value will be used if negative.
        area : float, optional
            The total area if one integrates coeff as a function of the time.
            If the area is negative, the pulse is flipped vertically
            (i.e. the pulse is multiplied by the sign of the area).

        Returns
        -------
        coeff, tlist :
            If the default window ``"shape"="rectangular"`` is used,
            both are float numbers.
            If Scipy window functions are used, both are a 1-dimensional numpy
            array with the same size.

        Notes
        -----
        If Scipy window functions are used, it is suggested to set
        ``Processor.pulse_mode`` to ``"continuous"``.
        Notice that finite number of sampling points will also make
        the total integral of the coefficients slightly deviate from ``area``.

        Examples
        --------
        .. plot::
            :context: reset

            from qutip_qip.compiler import GateCompiler
            import numpy as np
            compiler = GateCompiler()
            coeff, tlist= compiler.generate_pulse_shape(
                "hann",  # Scipy Hann window
                1000,  # 100 sampling point
                maximum=3.,
                # Notice that 2 pi is added to H by qutip solvers.
                area= 1.,
            )

        We can plot the generated pulse shape:

        .. plot::
            :context: close-figs

            import matplotlib.pyplot as plt
            plt.plot(tlist, coeff)
            plt.show()

        The pulse is normalized to fit the area. Notice that due to
        the finite number of sampling points, it is not exactly 1.

        .. testsetup::

            from qutip_qip.compiler import GateCompiler
            import numpy as np
            compiler = GateCompiler()
            coeff, tlist= compiler.generate_pulse_shape(
                "hann",  # Scipy Hann window
                1000,  # 100 sampling point
                maximum=3.,
                # Notice that 2 pi is added to H by qutip solvers.
                area= 1.,
            )

        .. doctest::

            >>> round(np.trapz(coeff, tlist), 2)
            1.0
        """
        coeff, tlist = _normalized_window(shape, num_samples)
        sign = np.sign(area)
        coeff *= np.abs(maximum) * sign
        tlist *= abs(area) / np.abs(maximum)
        return coeff, tlist


_default_window_t_max = {
    "boxcar": 1.0,
    "triang": 2.0,
    "blackman": 1.0 / 0.42,
    "hamming": 1.0 / 0.54,
    "hann": 2.0,
    "bartlett": 2.0,
    "flattop": 1.0 / 0.21557897160000217,
    "parzen": 1.0 / 0.375,
    "bohman": 1.0 / 0.4052847750978287,
    "blackmanharris": 1.0 / 0.35875003586900384,
    "nuttall": 1.0 / 0.36358193632191405,
    "barthann": 2.0,
    "cosine": np.pi / 2.0,
}


def _normalized_window(shape, num_samples):
    """
    Normalized SciPy window functions.
    The SciPy implementation only makes sure that it is maximum is 1.
    Here, we save a default t_max so that the integral is always 1.
    """
    if shape == "rectangular":
        return 1.0, 1.0
    t_max = _default_window_t_max.get(shape, None)
    if t_max is None:
        raise ValueError(f"Window function {shape} is not supported.")
    coeff = signal.windows.get_window(shape, num_samples)
    tlist = np.linspace(0, t_max, num_samples)
    return coeff, tlist
