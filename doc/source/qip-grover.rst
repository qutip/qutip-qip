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

    .. doctest::

        >>> from qutip_qip.algorithms.grover import grover, grover_oracle
        >>> n_qubits = 3
        >>> marked = [3, 5]
        >>> # 1. Create the Phase Oracle
        >>> oracle = grover_oracle(n_qubits, marked)
        >>> # 2. Build the Grover Circuit
        >>> # Since N=8 and M=2, the algorithm calculates optimal iterations automatically.
        >>> qc = grover(oracle, n_qubits, num_solutions=len(marked))

    We can then simulate the circuit and check the probability of success:

    .. doctest::

        >>> from qutip import basis, tensor
        >>> U_grover = qc.compute_unitary()
        >>> psi0 = tensor([basis(2, 0)] * n_qubits)
        >>> psi_final = U_grover * psi0
        >>> # Calculate probability of measuring state 3 or 5
        >>> prob = abs(psi_final.overlap(basis(8, 3)))**2 + abs(psi_final.overlap(basis(8, 5)))**2
        >>> print(f"Success Probability: {prob:.4f}")
        Success Probability: 1.0000

.. plot::
    :context: close-figs

    Grover's algorithm can also be visualized by plotting the histogram of measurement counts, showing the amplification of the marked states.

    .. code-block:: python

        from qiskit.visualization import plot_histogram
        # ... (after running simulation)
        # plot_histogram(counts)

References
==========

.. [1] L. K. Grover, "A fast quantum mechanical algorithm for database search," 
   Proceedings of the 28th Annual ACM Symposium on Theory of Computing (1996).