from collections.abc import Iterable
import warnings
from copy import deepcopy
from typing import Any, List, Tuple, Hashable

import numpy as np
from scipy.interpolate import CubicSpline

import qutip
from qutip import Qobj, QobjEvo, identity, tensor, mesolve, mcsolve
from ..operations import expand_operator, globalphase
from ..circuit import QubitCircuit
from ..noise import (
    Noise,
    RelaxationNoise,
    DecoherenceNoise,
    ControlAmpNoise,
    RandomNoise,
    process_noise,
)
from ..pulse import Pulse, Drift, _merge_qobjevo, _fill_coeff


__all__ = ["Processor"]


class Processor(object):
    """
    The noisy quantum device simulator using QuTiP dynamic solvers.
    It compiles quantum circuit into a Hamiltonian model and then
    simulate the time-evolution described by the master equation.

    .. note::

        This is an abstract class that includes the general API but
        has no concrete physical model implemented.
        In particular, it provides a series of low-level APIs that allow
        direct modification of the Hamiltonian model and control pulses,
        which can usually be achieved automatically using :obj:`.Model`
        and build-in workflows.
        They provides more flexibility but are not always the most
        elegant approaches.

    Parameters
    ----------
    num_qubits : int, optional
        The number of qubits.
        It replaces the old API ``N``.

    dims : list, optional
        The dimension of each component system.
        Default value is a qubit system of ``dim=[2,2,2,...,2]``.

    spline_kind : str, optional
        Type of the coefficient interpolation. Default is "step_func"
        Note that they have different requirements for the length of ``coeff``.

        -"step_func":
        The coefficient will be treated as a step function.
        E.g. ``tlist=[0,1,2]`` and ``coeff=[3,2]``, means that the coefficient
        is 3 in t=[0,1) and 2 in t=[2,3). It requires
        ``len(coeff)=len(tlist)-1`` or ``len(coeff)=len(tlist)``, but
        in the second case the last element of ``coeff`` has no effect.

        -"cubic": Use cubic interpolation for the coefficient. It requires
        ``len(coeff)=len(tlist)``

    model : :obj:`Model`
        Provide a predefined physical model of the simulated hardware.
        If other parameters, such as `t1` is given as input,
        it will overwrite those saved in :obj:`Processor.model.params`.

    t1 : float or list, optional
        Characterize the amplitude damping for each qubit.
        A list of size `num_qubits` or a float for all qubits.

    t2 : float or list, optional
        Characterize the total dephasing for each qubit.
        A list of size `num_qubits` or a float for all qubits.
    """

    def __init__(
        self,
        num_qubits=None,
        dims=None,
        spline_kind="step_func",
        model=None,
        N=None,
        t1=None,
        t2=None,
    ):
        num_qubits = num_qubits if num_qubits is not None else N
        if model is None:
            self.model = Model(num_qubits=num_qubits, dims=dims, t1=t1, t2=t2)
        else:
            self.model = model

        self.pulses = []
        # FIXME # Think about the handling of spline_kind.
        self.spline_kind = spline_kind

    @property
    def num_qubits(self):
        """
        Number of qubits (or subsystems). For backward compatibility.
        :type: int
        """
        return self.model.num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        self.model.num_qubits = value

    @property
    def dims(self):
        """
        The dimension of each component system.
        :type: list
        """
        return self.model.dims

    @dims.setter
    def dims(self, value):
        self.model.dims = value

    @property
    def t1(self):
        """
        Characterize the total amplitude damping of each qubit.
        :type: float or list
        """
        return self.model.params.get("t1", None)

    @t1.setter
    def t1(self, value):
        self.model.params["t1"] = value

    @property
    def t2(self):
        """
        Characterize the total dephasing for each qubit.
        :type: float or list
        """
        return self.model.params.get("t2", None)

    @t2.setter
    def t2(self, value):
        self.model.params["t2"] = value

    @property
    def params(self):
        """
        Hardware parameters.
        :type: dict
        """
        return self.model.params

    @property
    def noise(self):
        """.coverage"""
        return self.get_noise()

    @property
    def N(self):
        return self.num_qubits

    ####################################################################
    # Hamiltonian model
    def get_all_drift(self):
        """
        Get all the drift Hamiltonians.

        Returns
        -------
        drift_hamiltonian_list : list
            A list of drift Hamiltonians in the form of
            ``[(qobj, targets), ...]``.
        """
        return self.model.get_all_drift()

    @property
    def drift(self):
        """
        The drift Hamiltonian in the form ``[(qobj, targets), ...]``
        :type: list
        """
        return self.get_all_drift()

    def _get_drift_obj(self):
        """generate the Drift representation"""
        drift_obj = Drift()
        for qobj, targets in self.model.get_all_drift():
            num_qubits = len(qobj.dims[0])
            drift_obj.add_drift(qobj, targets)
        return drift_obj

    def _unify_targets(self, qobj, targets):
        if targets is None:
            targets = list(range(len(qobj.dims[0])))
        if not isinstance(targets, Iterable):
            targets = [targets]
        return targets

    def add_drift(self, qobj, targets=None, cyclic_permutation=False):
        """
        Add the drift Hamiltonian to the model.
        The drift Hamiltonians are intrinsic
        of the quantum system and cannot be controlled by an external field.

        Parameters
        ----------
        qobj : :class:`qutip.Qobj`
            The drift Hamiltonian.
        targets : list, optional
            The indices of the target qubits
            (or subquantum system of other dimensions).
        cyclic_permutation : bool, optional
            If true, the Hamiltonian will be added for all qubits,
            e.g. if ``targets=[0,1]``, and there are 2 qubits,
            The Hamiltonian will be added to the target qubits
            ``[0,1]``, ``[1,2]`` and ``[2,0]``.
        """
        targets = self._unify_targets(qobj, targets)
        if cyclic_permutation:
            for i in range(self.num_qubits):
                temp_targets = [(t + i) % self.num_qubits for t in targets]
                self.model._add_drift(qobj, temp_targets)
        else:
            self.model._add_drift(qobj, targets)

    def add_control(
        self, qobj, targets=None, cyclic_permutation=False, label=None
    ):
        """
        Add a control Hamiltonian to the model. The new control Hamiltonian
        is saved in the :obj:`.Processor.model` attributes.

        Parameters
        ----------
        qobj : :obj:`qutip.Qobj`
            The control Hamiltonian.

        targets : list, optional
            The indices of the target qubits
            (or composite quantum systems).

        cyclic_permutation : bool, optional
            If true, the Hamiltonian will be added for all qubits,
            e.g. if ``targets=[0,1]``, and there are 2 qubits,
            the Hamiltonian will be added to the target qubits
            ``[0,1]``, ``[1,2]`` and ``[2,0]``.

        label : str, optional
            The hashable label (name) of the control Hamiltonian.
            If ``None``,
            it will be set to the current number of
            control Hamiltonians in the system.

        Examples
        --------
        >>> import qutip
        >>> from qutip_qip.device import Processor
        >>> processor = Processor(1)
        >>> processor.add_control(qutip.sigmax(), 0, label="sx")
        >>> processor.get_control_labels()
        ['sx']
        >>> processor.get_control("sx") # doctest: +NORMALIZE_WHITESPACE
        (Quantum object: dims = [[2], [2]], shape = (2, 2),
        type = oper, isherm = True
        Qobj data =
        [[0. 1.]
        [1. 0.]], [0])
        """
        targets = self._unify_targets(qobj, targets)
        if label is None:
            label = len(self.model._controls)
        if cyclic_permutation:
            for i in range(self.num_qubits):
                temp_targets = [(t + i) % self.num_qubits for t in targets]
                temp_label = (label, tuple(temp_targets))
                self.model._add_control(temp_label, qobj, temp_targets)
        else:
            self.model._add_control(label, qobj, targets)

    def get_control(self, label):
        """
        Get the control Hamiltonian corresponding to the label.

        Parameters
        ----------
        label :
            A label that identifies the Hamiltonian.

        Returns
        -------
        control_hamiltonian : tuple
            The control Hamiltonian in the form of ``(qobj, targets)``.

        Examples
        --------
        >>> from qutip_qip.device import LinearSpinChain
        >>> processor = LinearSpinChain(1)
        >>> processor.get_control_labels()
        ['sx0', 'sz0']
        >>> processor.get_control('sz0') # doctest: +NORMALIZE_WHITESPACE
        (Quantum object: dims = [[2], [2]], shape = (2, 2),
        type = oper, isherm = True
        Qobj data =
        [[ 6.28318531  0.        ]
        [ 0.         -6.28318531]], 0)
        """
        return self.model.get_control(label)

    def get_control_labels(self):
        """
        Get a list of all available control Hamiltonians.

        Returns
        -------
        label_list : list
            A list of hashable objects each corresponds to
            an available control Hamiltonian.
        """
        return self.model.get_control_labels()

    def get_control_latex(self):
        """
        Get the latex string for each Hamiltonian.
        It is used in the method :meth:`.Processor.plot_pulses`.
        It is a list of dictionaries.
        In the plot, a different color will be used
        for each dictionary in the list.

        Returns
        -------
        nested_latex_str : list of dict
            E.g.: ``[{"sx": "\sigma_z"}, {"sy": "\sigma_y"}]``.
        """
        if hasattr(self.model, "get_control_latex"):
            return self.model.get_control_latex()
        labels = self.model.get_control_labels()
        return [{label: label for label in labels}]

    def get_noise(self):
        """
        Get a list of :obj:`.Noise` objects.

        Returns
        -------
        noise_list : list
            A list of :obj:`.Noise`.
        """
        if hasattr(self.model, "get_noise"):
            return self.model.get_noise()
        else:
            return []

    def add_noise(self, noise):
        """
        Add a noise object to the processor.

        Parameters
        ----------
        noise : :class:`.Noise`
            The noise object defined outside the processor.
        """
        if isinstance(noise, Noise):
            self.model._add_noise(noise)
        else:
            raise TypeError("Input is not a Noise object.")

    ####################################################################
    # Control coefficients
    @property
    def controls(self):
        """
        A list of the ideal control Hamiltonians in all saved pulses.
        Note that control Hamiltonians with no pulse will not be included.
        The order matches with :obj:`Processor.coeffs`
        """
        result = []
        for pulse in self.pulses:
            result.append(pulse.get_ideal_qobj(dims=self.dims))
        return result

    ctrls = controls

    @property
    def coeffs(self):
        """
        A list of ideal control coefficients for all saved pulses.
        The order matches with :obj:`Processor.controls`
        """
        if not self.pulses:
            return None
        coeffs_list = [pulse.coeff for pulse in self.pulses]
        return coeffs_list

    @coeffs.setter
    def coeffs(self, coeffs):
        self.set_coeffs(coeffs)

    def _generate_iterator_from_dict_or_list(self, value):
        if isinstance(value, dict):
            iterator = value.items()
        elif isinstance(value, (list, np.ndarray)):
            iterator = enumerate(value)
        else:
            raise ValueError("Wrong type.")
        return iterator

    def set_coeffs(self, coeffs):
        """
        Clear all the existing pulses and
        reset the coefficients for the control Hamiltonians.

        Parameters
        ----------
        coeffs: NumPy arrays, dict or list.
            - If it is a dict, it should be a map of
              the label of control Hamiltonians and the
              corresponding coefficients.
              Use :obj:`.Processor.get_control_labels()` to see the
              available Hamiltonians.
            - If it is a list of arrays or a 2D NumPy array,
              it is treated same to ``dict``, only that
              the pulse label is assumed to be integers from 0
              to ``len(coeffs)-1``.
        """
        self.clear_pulses()
        iterator = self._generate_iterator_from_dict_or_list(coeffs)
        for label, coeff in iterator:
            label = label
            ham, targets = self.model.get_control(label)
            self.add_pulse(
                Pulse(
                    ham,
                    targets,
                    coeff=coeffs[label],
                    spline_kind=self.spline_kind,
                    label=label,
                )
            )

    set_all_coeffs = set_coeffs

    def set_tlist(self, tlist):
        """
        Set the ``tlist`` for all existing pulses. It assumes that
        pulses all already added to the processor.
        To add pulses automatically, first use :obj:`Processor.set_coeffs`.

        Parameters
        ----------
        tlist: dict or list of NumPy arrays.
            If it is a dict, it should be a map between pulse label and
            the time sequences.
            If it is a list of arrays or a 2D NumPy array,
            each array will be associated
            to a pulse, following the order in the pulse list.
        """
        if isinstance(tlist, np.ndarray) and len(tlist.shape) == 1:
            for pulse in self.pulses:
                pulse.tlist = tlist
            return
        iterator = self._generate_iterator_from_dict_or_list(tlist)
        pulse_dict = self.get_pulse_dict()
        for pulse_label, value in iterator:
            self.pulses[pulse_dict[pulse_label]].tlist = value

    set_all_tlist = set_tlist

    def get_full_tlist(self, tol=1.0e-10):
        """
        Return the full tlist of the ideal pulses.
        If different pulses have different time steps,
        it will collect all the time steps in a sorted array.

        Returns
        -------
        full_tlist: array-like 1d
            The full time sequence for the ideal evolution.
        """
        full_tlist = [
            pulse.tlist for pulse in self.pulses if pulse.tlist is not None
        ]
        if not full_tlist:
            return None
        full_tlist = np.unique(np.sort(np.hstack(full_tlist)))
        # account for inaccuracy in float-point number
        full_tlist = np.concatenate(
            (full_tlist[:1], full_tlist[1:][np.diff(full_tlist) > tol])
        )
        return full_tlist

    def get_full_coeffs(self, full_tlist=None):
        """
        Return the full coefficients in a 2d matrix form.
        Each row corresponds to one pulse. If the `tlist` are
        different for different pulses, the length of each row
        will be the same as the `full_tlist` (see method
        `get_full_tlist`). Interpolation is used for
        adding the missing coefficients according to `spline_kind`.

        Returns
        -------
        coeffs: array-like 2d
            The coefficients for all ideal pulses.
        """
        # TODO add tests
        self._is_pulses_valid()
        if not self.pulses:
            return np.array((0, 0), dtype=float)
        if full_tlist is None:
            full_tlist = self.get_full_tlist()
        coeffs_list = []
        for pulse in self.pulses:
            if pulse.tlist is None and pulse.coeff is None:
                coeffs_list.append(np.zeros(len(full_tlist)))
                continue
            if not isinstance(pulse.coeff, (bool, np.ndarray)):
                raise ValueError(
                    "get_full_coeffs only works for "
                    "NumPy array or bool coeff."
                )
            if isinstance(pulse.coeff, bool):
                if pulse.coeff:
                    coeffs_list.append(np.ones(len(full_tlist)))
                else:
                    coeffs_list.append(np.zeros(len(full_tlist)))
                continue
            if self.spline_kind == "step_func":
                arg = {"_step_func_coeff": True}
                coeffs_list.append(
                    _fill_coeff(pulse.coeff, pulse.tlist, full_tlist, arg)
                )
            elif self.spline_kind == "cubic":
                coeffs_list.append(
                    _fill_coeff(pulse.coeff, pulse.tlist, full_tlist, {})
                )
            else:
                raise ValueError("Unknown spline kind.")
        return np.array(coeffs_list)

    def save_coeff(self, file_name, inctime=True):
        """
        Save a file with the control amplitudes in each timeslot.

        Parameters
        ----------
        file_name: string
            Name of the file.

        inctime: bool, optional
            True if the time list should be included in the first column.
        """
        self._is_pulses_valid()
        coeffs = np.array(self.get_full_coeffs())

        if not all([isinstance(pulse.label, str) for pulse in self.pulses]):
            raise NotImplementedError("Only string labels are supported.")
        header = ";".join([str(pulse.label) for pulse in self.pulses])
        if inctime:
            shp = coeffs.T.shape
            data = np.empty((shp[0], shp[1] + 1), dtype=np.float64)
            data[:, 0] = self.get_full_tlist()
            data[:, 1:] = coeffs.T
            header = ";" + header
        else:
            data = coeffs.T

        np.savetxt(
            file_name, data, delimiter="\t", fmt="%1.16f", header=header
        )

    def read_coeff(self, file_name, inctime=True):
        """
        Read the control amplitudes matrix and time list
        saved in the file by `save_amp`.

        Parameters
        ----------
        file_name: string
            Name of the file.

        inctime: bool, optional
            True if the time list in included in the first column.

        Returns
        -------
        tlist: array_like
            The time list read from the file.

        coeffs: array_like
            The pulse matrix read from the file.
        """
        f = open(file_name)
        header = f.readline()
        label_list = header[2:-1].split(";")
        f.close()

        data = np.loadtxt(file_name, delimiter="\t")
        if not inctime:
            coeffs = data.T
        else:
            tlist = data[:, 0]
            coeffs = data[:, 1:].T
            label_list = label_list[1:]
        coeffs = {label: coeffs[i] for i, label in enumerate(label_list)}
        self.set_coeffs(coeffs)
        if not inctime:
            return coeffs
        else:
            self.set_tlist(tlist)
            return self.get_full_tlist, coeffs

    ####################################################################
    # Pulse
    def add_pulse(self, pulse):
        """
        Add a new pulse to the device.

        Parameters
        ----------
        pulse : :class:`.Pulse`
            `Pulse` object to be added.
        """
        if isinstance(pulse, Pulse):
            if pulse.spline_kind is None:
                pulse.spline_kind = self.spline_kind
            self.pulses.append(pulse)
        else:
            raise ValueError("Invalid input, pulse must be a Pulse object")

    def remove_pulse(self, indices=None, label=None):
        """
        Remove the control pulse with given indices.

        Parameters
        ----------
        indices: int or list of int
            The indices of the control Hamiltonians to be removed.
        label: str
            The label of the pulse
        """
        if indices is not None:
            if not isinstance(indices, Iterable):
                indices = [indices]
            indices.sort(reverse=True)
            for ind in indices:
                del self.pulses[ind]
        else:
            for ind, pulse in enumerate(self.pulses):
                if pulse.label == label:
                    del self.pulses[ind]

    def clear_pulses(self):
        self.pulses = []

    def _is_pulses_valid(self):
        """
        Check if the pulses are in the correct shape.

        Returns: bool
            If they are valid or not
        """
        for i, pulse in enumerate(self.pulses):
            if pulse.coeff is None or isinstance(pulse.coeff, bool):
                # constant pulse
                continue
            if pulse.tlist is None:
                raise ValueError(
                    "Pulse id={} is invalid. "
                    "Please define a tlist for the pulse.".format(i)
                )
            if pulse.tlist is not None and pulse.coeff is None:
                raise ValueError(
                    "Pulse id={} is invalid. "
                    "Please define a coeff for the pulse.".format(i)
                )
            coeff_len = len(pulse.coeff)
            tlist_len = len(pulse.tlist)
            if pulse.spline_kind == "step_func":
                if coeff_len == tlist_len - 1 or coeff_len == tlist_len:
                    pass
                else:
                    raise ValueError(
                        "The length of tlist and coeff of the pulse "
                        "labelled {} is invalid. "
                        "It's either len(tlist)=len(coeff) or "
                        "len(tlist)-1=len(coeff) for coefficients "
                        "as step function".format(i)
                    )
            else:
                if coeff_len == tlist_len:
                    pass
                else:
                    raise ValueError(
                        "The length of tlist and coeff of the pulse "
                        "labelled {} is invalid. "
                        "It should be either len(tlist)=len(coeff)".format(i)
                    )
        return True

    def get_pulse_dict(self):
        label_list = {}
        for i, pulse in enumerate(self.pulses):
            if pulse.label is not None:
                label_list[pulse.label] = i
        return label_list

    def find_pulse(self, pulse_name):
        pulse_dict = self.get_pulse_dict()
        if isinstance(pulse_name, int):
            return self.pulses[pulse_name]
        else:
            try:
                return self.pulses[pulse_dict[pulse_name]]
            except (KeyError):
                raise KeyError(
                    "Pulse name {} undefined. "
                    "Please define it in the attribute "
                    "`pulse_dict`.".format(pulse_name)
                )

    @property
    def pulse_mode(self):
        """
        If the given pulse is going to be interpreted as
        "continuous" or "discrete".

        :type: str
        """
        if self.spline_kind == "step_func":
            return "discrete"
        elif self.spline_kind == "cubic":
            return "continuous"
        else:
            raise ValueError("Saved spline_kind not understood.")

    @pulse_mode.setter
    def pulse_mode(self, mode):
        if mode == "discrete":
            spline_kind = "step_func"
        elif mode == "continuous":
            spline_kind = "cubic"
        else:
            raise ValueError(
                "Pulse mode must be either discrete or continuous."
            )

        self.spline_kind = spline_kind
        for pulse in self.pulses:
            pulse.spline_kind = spline_kind

    def plot_pulses(
        self,
        title=None,
        figsize=(12, 6),
        dpi=None,
        show_axis=False,
        rescale_pulse_coeffs=True,
        num_steps=1000,
        pulse_labels=None,
        use_control_latex=True,
    ):
        """
        Plot the ideal pulse coefficients.

        Parameters
        ----------
        title: str, optional
            Title for the plot.

        figsize: tuple, optional
            The size of the figure.

        dpi: int, optional
            The dpi of the figure.

        show_axis: bool, optional
            If the axis are shown.

        rescale_pulse_coeffs: bool, optional
            Rescale the hight of each pulses.

        num_steps: int, optional
            Number of time steps in the plot.

        pulse_labels: list of dict, optional
            A map between pulse labels and the labels shown in the y axis.
            E.g. ``[{"sx": "sigmax"}]``.
            Pulses in each dictionary will get a different color.
            If not given and ``use_control_latex==False``,
            the string label defined in each :obj:`.Pulse` is used.

        use_control_latex: bool, optional
            Use labels defined in ``Processor.model.get_control_latex``.

        pulse_labels: list of dict, optional
            A map between pulse labels and the labels shown on the y axis.
            E.g. ``["sx", "sigmax"]``.
            If not given and ``use_control_latex==False``,
            the string label defined in each :obj:`.Pulse` is used.

        use_control_latex: bool, optional
            Use labels defined in ``Processor.model.get_control_latex``.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The `Figure` object for the plot.

        axis: list of ``matplotlib.axes._subplots.AxesSubplot``
            The axes for the plot.

        Notes
        -----
        :meth:.Processor.plot_pulses` only works for array_like coefficients.
        """
        if hasattr(self, "get_operators_labels"):
            warnings.warn(
                "Using the get_operators_labels to provide labels "
                "for plotting is deprecated. "
                "Please use get_control_latex instead."
            )
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # choose labels
        if pulse_labels is None:
            if use_control_latex and not hasattr(
                self.model, "get_control_latex"
            ):
                warnings.warn(
                    "No method get_control_latex defined in the model. "
                    "Switch to using the labels defined in each pulse."
                    "Set use_control_latex=False to turn off the warning."
                )
            if use_control_latex:  # use control labels in the model
                control_labels = deepcopy(self.get_control_latex())
                pulse_labels = control_labels
            else:
                pulse_labels = [
                    {pulse.label: pulse.label for pulse in self.pulses}
                ]

        # If it is a nested list instead of a list of dict, we assume that
        if isinstance(pulse_labels[0], list):
            for ind, pulse_group in enumerate(pulse_labels):
                pulse_labels[ind] = {
                    i: latex for i, latex in enumerate(pulse_group)
                }

        # create a axis for each pulse
        fig = plt.figure(figsize=figsize, dpi=dpi)
        grids = gridspec.GridSpec(sum([len(d) for d in pulse_labels]), 1)
        grids.update(wspace=0.0, hspace=0.0)

        tlist = np.linspace(0.0, self.get_full_tlist()[-1], num_steps)
        dt = tlist[1] - tlist[0]

        # make sure coeffs start and end with zero, for ax.fill
        tlist = np.hstack(([-dt * 1.0e-20], tlist, [tlist[-1] + dt * 1.0e-20]))
        coeffs = []
        for pulse in self.pulses:
            coeffs.append(_pulse_interpolate(pulse, tlist))

        pulse_ind = 0
        axis = []
        for i, label_group in enumerate(pulse_labels):
            for j, (label, latex_str) in enumerate(label_group.items()):
                try:
                    pulse = self.find_pulse(label)
                    coeff = _pulse_interpolate(pulse, tlist)
                except KeyError:
                    coeff = np.zeros(tlist.shape)
                grid = grids[pulse_ind]
                ax = plt.subplot(grid)
                axis.append(ax)
                ax.fill(tlist, coeff, color_list[i], alpha=0.7)
                ax.plot(tlist, coeff, color_list[i])
                if rescale_pulse_coeffs:
                    ymax = np.max(np.abs(coeff)) * 1.1
                else:
                    ymax = np.max(np.abs(coeffs)) * 1.1
                if ymax != 0.0:
                    ax.set_ylim((-ymax, ymax))

                # disable frame and ticks
                if not show_axis:
                    ax.set_xticks([])
                    ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_yticks([])
                ax.set_ylabel(latex_str, rotation=0)
                pulse_ind += 1
                if i == 0 and j == 0 and title is not None:
                    ax.set_title(title)
        fig.tight_layout()
        return fig, axis

    ####################################################################
    # Simulation API and utilities
    def get_noisy_pulses(self, device_noise=False, drift=False):
        """
        It takes the pulses defined in the `Processor` and
        adds noise according to `Processor.noise`. It does not modify the
        pulses saved in `Processor.pulses` but returns a new list.
        The length of the new list of noisy pulses might be longer
        because of drift Hamiltonian and device noise. They will be
        added to the end of the pulses list.

        Parameters
        ----------
        device_noise: bool, optional
            If true, include pulse independent noise such as single qubit
            Relaxation. Default is False.
        drift: bool, optional
            If true, include drift Hamiltonians. Default is False.

        Returns
        -------
        noisy_pulses : list of :class:`.Drift`
            A list of noisy pulses.
        """
        pulses = deepcopy(self.pulses)
        noisy_pulses = process_noise(
            pulses,
            self.noise,
            self.dims,
            t1=self.t1,
            t2=self.t2,
            device_noise=device_noise,
        )
        if drift:
            drift_obj = self._get_drift_obj()
            noisy_pulses += [drift_obj]
        return noisy_pulses

    def get_qobjevo(self, args=None, noisy=False):
        """
        Create a :class:`qutip.QobjEvo` representation of the evolution.
        It calls the method :meth:`.Processor.get_noisy_pulses` and create
        the `QobjEvo` from it.

        Parameters
        ----------
        args: dict, optional
            Arguments for :class:`qutip.QobjEvo`
        noisy: bool, optional
            If noise are included. Default is False.

        Returns
        -------
        qobjevo : :class:`qutip.QobjEvo`
            The :class:`qutip.QobjEvo` representation of the unitary evolution.
        c_ops: list of :class:`qutip.QobjEvo`
            A list of lindblad operators is also returned.
            if ``noisy==False``,
            it is always an empty list.
        """
        # TODO test it for non array-like coeff
        # check validity
        self._is_pulses_valid()

        if args is None:
            args = {}
        else:
            args = args
        # set step function

        if not noisy:
            dynamics = self.pulses
        else:
            dynamics = self.get_noisy_pulses(device_noise=True, drift=True)

        qu_list = []
        c_ops = []
        for pulse in dynamics:
            if noisy:
                qu, new_c_ops = pulse.get_noisy_qobjevo(dims=self.dims)
                c_ops += new_c_ops
            else:
                qu = pulse.get_ideal_qobjevo(dims=self.dims)
            qu_list.append(qu)

        final_qu = _merge_qobjevo(qu_list)
        final_qu.args.update(args)

        # bring all c_ops to the same tlist, won't need it in QuTiP 5
        temp = []
        for c_op in c_ops:
            temp.append(_merge_qobjevo([c_op], final_qu.tlist))
        c_ops = temp

        if noisy:
            return final_qu, c_ops
        else:
            return final_qu, []

    def run_analytically(self, init_state=None, qc=None):
        """
        Simulate the state evolution under the given `qutip.QubitCircuit`
        with matrice exponentiation. It will calculate the propagator
        with matrix exponentiation and return a list of :class:`qutip.Qobj`.
        This method won't include noise or collpase.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`, optional
            Takes the quantum circuit to be implemented. If not given, use
            the quantum circuit saved in the processor by ``load_circuit``.

        init_state : :class:`qutip.Qobj`, optional
            The initial state of the qubits in the register.

        Returns
        -------
        U_list: list
            A list of propagators obtained for the physical implementation.
        """
        if init_state is not None:
            U_list = [init_state]
        else:
            U_list = []
        tlist = self.get_full_tlist()
        coeffs = self.get_full_coeffs()

        # Compute drift Hamiltonians
        H_drift = 0
        drift = self._get_drift_obj()
        for drift_ham in drift.drift_hamiltonians:
            H_drift += drift_ham.get_qobj(self.dims)

        # Compute control Hamiltonians
        for n in range(len(tlist) - 1):
            H = H_drift + sum(
                [
                    coeffs[m, n] * self.pulses[m].get_ideal_qobj(self.dims)
                    for m in range(len(self.pulses))
                ]
            )
            dt = tlist[n + 1] - tlist[n]
            U = (-1j * H * dt).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        try:  # correct_global_phase are defined for ModelProcessor
            if self.correct_global_phase and self.global_phase != 0:
                U_list.append(
                    globalphase(self.global_phase, N=self.num_qubits)
                )
        except AttributeError:
            pass

        return U_list

    def run(self, qc=None):
        """
        Calculate the propagator of the evolution by matrix exponentiation.
        This method won't include noise or collpase.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`, optional
            Takes the quantum circuit to be implemented. If not given, use
            the quantum circuit saved in the processor by `load_circuit`.

        Returns
        -------
        U_list: list
            The propagator matrix obtained from the physical implementation.
        """
        if qc:
            self.load_circuit(qc)
        return self.run_analytically(qc=qc, init_state=None)

    def run_state(
        self,
        init_state=None,
        analytical=False,
        states=None,
        noisy=True,
        solver="mesolve",
        **kwargs
    ):
        """
        If `analytical` is False, use :func:`qutip.mesolve` to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as keyword arguments.

        If `analytical` is True, calculate the propagator
        with matrix exponentiation and return a list of matrices.
        Noise will be neglected in this option.

        Parameters
        ----------
        init_state : :class:`qutip.Qobj`
            Initial density matrix or state vector (ket).

        analytical: bool
            If True, calculate the evolution with matrices exponentiation.

        states : :class:`qutip.Qobj`, optional
            Old API, same as init_state.

        solver: str
            "mesolve" or "mcsolve",
            for :func:`~qutip.mesolve` and :func:`~qutip.mcsolve`.

        noisy: bool
            Include noise or not.

        **kwargs
            Keyword arguments for the qutip solver.
            E.g `tlist` for time points for recording
            intermediate states and expectation values;
            `args` for the solvers and `qutip.QobjEvo`.

        Returns
        -------
        evo_result : :class:`qutip.Result`
            If ``analytical`` is False,  an instance of the class
            :class:`qutip.Result` will be returned.

            If ``analytical`` is True, a list of matrices representation
            is returned.
        """
        if states is not None:
            warnings.warn(
                "states will be deprecated and replaced by init_state",
                DeprecationWarning,
            )
        if init_state is None and states is None:
            raise ValueError("Qubit state not defined.")
        elif init_state is None:
            # just to keep the old parameters `states`,
            # it is replaced by init_state
            init_state = states
        if analytical:
            if kwargs or self.noise:
                raise warnings.warn(
                    "Analytical matrices exponentiation"
                    "does not process noise or"
                    "any keyword arguments."
                )
            return self.run_analytically(init_state=init_state)

        # kwargs can not contain H
        if "H" in kwargs:
            raise ValueError(
                "`H` is already specified by the processor "
                "and can not be given as a keyword argument"
            )

        # construct qobjevo for unitary evolution
        if "args" in kwargs:
            noisy_qobjevo, sys_c_ops = self.get_qobjevo(
                args=kwargs["args"], noisy=noisy
            )
        else:
            noisy_qobjevo, sys_c_ops = self.get_qobjevo(noisy=noisy)

        # add collpase operators into kwargs
        if "c_ops" in kwargs:
            if isinstance(kwargs["c_ops"], (Qobj, QobjEvo)):
                kwargs["c_ops"] += [kwargs["c_ops"]] + sys_c_ops
            else:
                kwargs["c_ops"] += sys_c_ops
        else:
            kwargs["c_ops"] = sys_c_ops

        # choose solver:
        if "tlist" in kwargs:
            tlist = kwargs["tlist"]
            del kwargs["tlist"]
        else:
            tlist = noisy_qobjevo.tlist
        if solver == "mesolve":
            evo_result = mesolve(
                H=noisy_qobjevo, rho0=init_state, tlist=tlist, **kwargs
            )
        elif solver == "mcsolve":
            evo_result = mcsolve(
                H=noisy_qobjevo, psi0=init_state, tlist=tlist, **kwargs
            )

        return evo_result

    def load_circuit(self, qc):
        """
        Translate an :class:`.QubitCircuit` to its
        corresponding Hamiltonians. (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        (Defined in subclasses)
        """
        return U


def _pulse_interpolate(pulse, tlist):
    """
    A function that calls Scipy interpolation routine. Used for plotting.
    """
    if pulse.tlist is None and pulse.coeff is None:
        coeff = np.zeros(len(tlist))
        return coeff
    if isinstance(pulse.coeff, bool):
        if pulse.coeff:
            coeff = np.ones(len(tlist))
        else:
            coeff = np.zeros(len(tlist))
        return coeff
    coeff = pulse.coeff
    if len(coeff) == len(pulse.tlist) - 1:  # for discrete pulse
        coeff = np.concatenate([coeff, [0]])

    from scipy import interpolate

    if pulse.spline_kind == "step_func":
        kind = "previous"
    else:
        kind = "cubic"
    inter = interpolate.interp1d(
        pulse.tlist, coeff, kind=kind, bounds_error=False, fill_value=0.0
    )
    return inter(tlist)


class Model:
    """
    Template class for a physical model representing quantum hardware.
    The concrete model class does not have to inherit from this,
    as long as the following methods are defined.

    Parameters
    ----------
    num_The number of qubits
        The number of qubits.
    dims : list, optional
        The dimension of each component system.
        Default value is a qubit system of ``dim=[2,2,2,...,2]``.
    **params :
        Hardware parameters for the model.

    Attributes
    ----------
    num_The number of qubits
        The number of qubits.
    dims : list, optional
        The dimension of each component system.
    params : dict
        Hardware parameters for the model.
    """

    def __init__(self, num_qubits, dims=None, **params):
        self.num_qubits = num_qubits if num_qubits is not None else N
        self.dims = dims if dims is not None else num_qubits * [2]
        self.params = deepcopy(params)
        self._controls = {}
        self._drift = []
        self._noise = []

    def get_all_drift(self) -> List[Tuple[Qobj, List[int]]]:
        """
        Get all the drift Hamiltonians.

        Returns
        -------
        drift_hamiltonian_list : list
            A list of drift Hamiltonians in the form of
            ``[(qobj, targets), ...]``.
        """
        return self._drift

    def get_control(self, label: Hashable) -> Tuple[Qobj, List[int]]:
        """
        Get the control Hamiltonian corresponding to the label.

        Parameters
        ----------
        label : hashable object
            A label that identifies the Hamiltonian.

        Returns
        -------
        control_hamiltonian : tuple
            The control Hamiltonian in the form of ``(qobj, targets)``.
        """
        if hasattr(self, "_old_index_label_map"):
            _old_index_label_map = self._old_index_label_map
            if isinstance(label, int):
                label = _old_index_label_map[label]
        return self._controls[label]

    def get_control_labels(self) -> List[Hashable]:
        """
        Get a list of all available control Hamiltonians.
        Optional, required only when plotting the pulses or
        using the optimal control algorithm.

        Returns
        -------
        label_list : list of hashable objects
            A list of hashable objects each corresponds to
            an available control Hamiltonian.
        """
        return list(self._controls.keys())

    def get_noise(self) -> List[Noise]:
        """
        Get a list of :obj:`.Noise` objects.
        Single qubit relaxation (T1, T2) are not included here.
        Optional method.

        Returns
        -------
        noise_list : list
            A list of :obj:`.Noise`.
        """
        if not hasattr(self, "_noise"):
            return []
        return self._noise

    def _add_drift(self, qobj, targets):
        if not hasattr(self, "_drift"):
            raise NotImplementedError(
                "The model does not support adding drift."
            )
        self._drift.append((qobj, targets))

    def _add_control(self, label, qobj, targets):
        if not hasattr(self, "_controls"):
            raise NotImplementedError(
                "The model does not support adding controls."
            )
        self._controls[label] = (qobj, targets)

    def _add_noise(self, noise):
        if not hasattr(self, "_noise"):
            raise NotImplementedError(
                "The model does not support adding noise objects."
            )
        self._noise.append(noise)
