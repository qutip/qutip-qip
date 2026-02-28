.. _qip_deutsch_jozsa:

**************************
Deutsch-Jozsa Algorithm
**************************

The Deutsch-Jozsa algorithm [1]_ was one of the first examples of a quantum algorithm that is exponentially faster than any deterministic classical algorithm. 
It determines whether a hidden boolean function $f: \{0,1\}^n \rightarrow \{0,1\}$ is **constant** (outputs the same value for all inputs) or **balanced** (outputs 0 for exactly half of the inputs and 1 for the other half).

Overview
========

While a classical computer requires $2^{n-1} + 1$ queries to the function to be certain of its nature, a quantum computer can determine the property in a **single query**. 
In ``qutip-qip``, the implementation follows these stages:

* **State Preparation**: All $n$ input qubits are placed in a uniform superposition using Hadamard gates, while an auxiliary (ancilla) qubit is initialized to the :math:`|-\rangle` state.
* **The Oracle**: A quantum transformation that encodes the function $f(x)$. It applies the transformation :math:`|x\rangle |y\rangle \rightarrow |x\rangle |y \oplus f(x)\rangle`.
* **Interference**: A final set of Hadamard gates is applied to the input register to cause destructive interference on all states except :math:`|0\rangle^{\otimes n}` if the function is constant. [cite: 8]


Constructing the Algorithm
==========================

In this module, the :func:`.deutsch_jozsa` function builds the circuit, and the utility :func:`.dj_oracle` helps generate test oracles.

The :func:`.deutsch_jozsa` function requires the following:

====================  ==================================================
Property                           Description
====================  ==================================================
``num_qubits``        The number of input qubits (excluding the auxiliary qubit).
``oracle``            A :class:`.QubitCircuit` or :class:`.Gate` representing the mystery function.
====================  ==================================================

Example: Classifying a Balanced Function
========================================

In this example, we generate a random balanced oracle for a 3-qubit system and use the algorithm to identify it.

.. doctest::

    >>> from qutip_qip.algorithms.deutsch_jozsa import deutsch_jozsa, dj_oracle
    >>> n_qubits = 3
    >>> # 1. Create a mystery balanced oracle
    >>> oracle = dj_oracle(n_qubits, case="balanced")
    >>> # 2. Build the Deutsch-Jozsa circuit
    >>> qc = deutsch_jozsa(n_qubits, oracle)

We can simulate the circuit and analyze the output state. If the function is balanced, the measurement of the input register will yield a non-zero result.

.. doctest::

    >>> from qutip import basis, tensor
    >>> U_dj = qc.compute_unitary()
    >>> # The algorithm starts with all qubits in |0>
    >>> psi0 = tensor([basis(2, 0)] * (n_qubits + 1))
    >>> final_state = U_dj * psi0
    >>> # Check probability of the input register being |000>
    >>> # We sum the probabilities for both possible states of the ancilla qubit.
    >>> prob_zero = abs(final_state.overlap(tensor([basis(2, 0)] * 4)))**2 + \
    ...             abs(final_state.overlap(tensor([basis(2, 0)] * 3 + [basis(2, 1)])))**2
    >>> print(f"Probability of measuring |000>: {prob_zero:.4f}")
    Probability of measuring |000>: 0.0000

.. plot::
    :context: close-figs

    from qiskit.visualization import plot_histogram
    # In a balanced case, the histogram would show zero counts for the '000' state.
    # plot_histogram(counts)

References
==========

.. [1] Deutsch, D. and Jozsa, R., "Rapid solution of problems by quantum computation," 
   Proceedings of the Royal Society of London. Series A: Mathematical and Physical Sciences, 439(1907), pp.553-558 (1992).