from qutip import Qobj


class CircuitResult:
    """
    Result of a quantum circuit simulation.
    """

    def __init__(self, final_states, probabilities, cbits=None):
        """
        Store result of CircuitSimulator.

        Parameters
        ----------
        final_states: list of Qobj.
            List of output kets or density matrices.

        probabilities: list of float.
            List of probabilities of obtaining each output state.

        cbits: list of list of int, optional
            List of cbits for each output.
        """

        if isinstance(final_states, Qobj) or final_states is None:
            self.final_states = [final_states]
            self.probabilities = [probabilities]
            if cbits:
                self.cbits = [cbits]
        else:
            inds = list(
                filter(
                    lambda x: final_states[x] is not None,
                    range(len(final_states)),
                )
            )
            self.final_states = [final_states[i] for i in inds]
            self.probabilities = [probabilities[i] for i in inds]
            if cbits:
                self.cbits = [cbits[i] for i in inds]

    def get_final_states(self, index=None):
        """
        Return list of output states.

        Parameters
        ----------
        index: int
            Indicates i-th state to be returned.

        Returns
        -------
        final_states: Qobj or list of Qobj.
            List of output kets or density matrices.
        """

        if index is not None:
            return self.final_states[index]
        return self.final_states

    def get_probabilities(self, index=None):
        """
        Return list of probabilities corresponding to the output states.

        Parameters
        ----------
        index: int
            Indicates i-th probability to be returned.

        Returns
        -------
        probabilities: float or list of float
            Probabilities associated with each output state.
        """

        if index is not None:
            return self.probabilities[index]
        return self.probabilities

    def get_cbits(self, index=None):
        """
        Return list of classical bit outputs corresponding to the results.

        Parameters
        ----------
        index: int
            Indicates i-th output, probability pair to be returned.

        Returns
        -------
        cbits: list of int or list of list of int
            list of classical bit outputs
        """

        if index is not None:
            return self.cbits[index]
        return self.cbits

    def plot_histogram(self, fig=None, ax=None, color="#1f77b4"):
        """
        Plot a histogram of the measurement outcomes and their probabilities.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure object for the histogram plot. If not provided, a new figure will be created.

        ax : matplotlib.axes.Axes, optional
            The axes object for the histogram plot. If not provided, a new axes will be created.

        color : str, optional
            Bar color for the histogram. Default is '#1f77b4'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object for the histogram plot.

        ax : matplotlib.axes.Axes
            The axes object for the histogram plot.
        """
        import matplotlib.pyplot as plt

        num_cbits = len(self.cbits[0])
        plot_dict = {f"{i:0{num_cbits}b}": 0.0 for i in range(1 << num_cbits)}

        for cbits, prob in zip(self.cbits, self.probabilities):
            binary = "".join(str(b) for b in cbits)
            plot_dict[binary] += prob

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.bar(
            plot_dict.keys(),
            plot_dict.values(),
            color=color,
            edgecolor="black",
            zorder=3,
        )
        ax.set_xlabel("Classical Register State")
        ax.set_ylabel("Probability")
        ax.set_title("Measurement Histogram")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

        return fig, ax
