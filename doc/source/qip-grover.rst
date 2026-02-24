.. _qip_grover:

**************************
Grover's Search Algorithm
**************************

Grover's algorithm [1]_ provides a quadratic speedup for searching an unstructured database of size $N$. 
It works by iteratively rotating a quantum state vector toward a "marked" solution state using a phase oracle and a diffusion operator.

Overview
========

Grover's algorithm is a fundamental quantum algorithm that finds the unique input to a black-box function that produces a particular output value. In ``qutip-qip``, the implementation consists of two main components:

* **The Oracle**: A phase oracle that flips the sign of the amplitude for the "marked" or "target" states.
* **The Diffusion Operator**: Also known as the "inversion about the mean" operator, which amplifies the probability of measuring the marked states.

The algorithm requires an optimal number of iterations, approximately $\frac{\pi}{4}\sqrt{\frac{N}{M}}$, where $N=2^n$ is the size of the search space and $M$ is the number of solutions.

Constructing the Algorithm
==========================

In this module, you can construct the full search circuit using the :func:`.grover` function. To make the process easier, a utility function :func:`.grover_oracle` is provided to generate standard phase-flip oracles.

The :func:`.grover` function requires the following:

====================  ==================================================
Property                           Description
====================  ==================================================
``oracle``            A :class:`.QubitCircuit`, :class:`.Gate`, or :class:`~.Qobj` representing the phase oracle.
``qubits``            List of qubit indices (or integer count) to run the search on.
``num_solutions``     **Mandatory** integer representing the number of marked states $M$.
``num_iterations``    Optional integer for manual control over the rotation count.
====================  ==================================================

Example: Searching for Multiple Targets
=======================================

Let's simulate a search on 3 qubits ($N=8$) where two states are marked: $|011\rangle$ (index 3) and $|101\rangle$ (index 5).

.. plot::
    :context: close-figs

    First, we import the necessary components and define our search space.

    >>> import matplotlib.pyplot as plt
    >>> from qiskit.visualization import plot_histogram
    >>> from qutip import basis, tensor
    >>> from qutip_qip.algorithms.grover import grover, grover_oracle
    >>> n_qubits = 3
    >>> marked = [3, 5]

    We then use the utility function :func:`.grover_oracle` to create a phase-flip oracle for our target states[cite: 8].

    >>> oracle = grover_oracle(n_qubits, marked)

    Using the :func:`.grover` function, we construct the full circuit. Since we provide the mandatory ``num_solutions``, the algorithm automatically calculates the optimal number of iterations[cite: 7, 11, 15].

    >>> qc = grover(oracle, n_qubits, num_solutions=len(marked))

    We can now simulate the circuit to obtain the final state vector[cite: 17].

    >>> U_grover = qc.compute_unitary()
    >>> psi0 = tensor([basis(2, 0)] * n_qubits)
    >>> psi_final = U_grover * psi0

    To visualize the results, we calculate the measurement probabilities for all possible states and plot them[cite: 17, 18].

    >>> probabilities = {
    ...     format(i, f'0{n_qubits}b'): float(abs(psi_final.overlap(basis(2**n_qubits, i)))**2)
    ...     for i in range(2**n_qubits)
    ... }
    >>> _ = plot_histogram(probabilities, title="Grover Search Results (2 Targets)")

References
==========

.. [1] L. K. Grover, "A fast quantum mechanical algorithm for database search," 
   Proceedings of the 28th Annual ACM Symposium on Theory of Computing (1996).