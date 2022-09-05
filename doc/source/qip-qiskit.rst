******************************
Qiskit
******************************

Overview
===============

This submodule provides an interface to simulate circuits made in qiskit.

Gate-level simulation on qiskit circuits is possible with :class:`.QiskitCircuitSimulator`. Pulse-level simulation is possible with :class:`.QiskitPulseSimulator` which supports simulation using the :class:`.LinearSpinChain`, :class:`.CircularSpinChain` and :class:`.DispersiveCavityQED` pulse processors.

Running a qiskit circuit with qutip_qip
==========================================

After constructing a circuit in qiskit, either of the qutip_qip based backends (:class:`.QiskitCircuitSimulator` and :class:`.QiskitPulseSimulator`) can be used to run that circuit.

Example
--------

Let's try constructing and simulating a qiskit circuit.

We define a simple circuit as follows:

.. code-block::

    from qiskit import QuantumCircuit
    
    circ = QuantumCircuit(2,2)
    circ.h(0)
    circ.h(1)
    
    circ.measure(0,0)
    circ.measure(1,1)

Let's run this on the :class:`.QiskitCircuitSimulator` backend:

.. code-block::

    from qutip_qip.qiskit.provider import QiskitCircuitSimulator

    backend = QiskitCircuitSimulator()
    job = backend.run(circ)
    result = job.result()
    
>>> from qiskit.visualization import plot_histogram
>>> plot_histogram(result.get_counts())

.. image:: /figures/qiskit-gate-level-plot.png
    :alt: probabilities plot 

Now, let's run the same circuit on :class:`.QiskitPulseSimulator`.

While using a pulse processor, we define the circuit without measurements:

.. code-block:: 

    pulse_circ = QuantumCircuit(2,2)
    pulse_circ.h(0)
    pulse_circ.h(1)

To use the :class:`.QiskitPulseSimulator` backend, we need to define the processor on which we want to run the circuit:

.. code-block::

    from qutip_qip.device import LinearSpinChain
    processor = LinearSpinChain(num_qubits=2)

Now that we defined our processor (:class:`.LinearSpinChain` in this case), we can use it to perform the simulation: 

.. code-block::

    from qutip_qip.qiskit.provider import QiskitPulseSimulator

    pulse_backend = QiskitPulseSimulator(processor)
    pulse_job = pulse_backend.run(pulse_circ)
    pulse_result = pulse_job.result()

>>> plot_histogram(pulse_result.get_counts())

.. image:: /figures/qiskit-pulse-level-plot.png
    :alt: probabilities plot

Configurable Options
========================

Qiskit's interface allows us to provide some options like ``shots`` while running a circuit on a backend. We also have provided some options for the qutip_qip backends.

``shots``
-------------
(Available for both: :class:`.QiskitCircuitSimulator` and :class:`.QiskitPulseSimulator`)
``shots`` is the number of times measurements are sampled from the simulation result. By default it is set to ``1024``.

``allow_custom_gate``
-----------------------
(Only available for :class:`.QiskitCircuitSimulator`)
``allow_custom_gate``, when set to ``False``, does not allowing simulating circuits that have user-defined gates; it will throw an error in that case. By default, it is set to ``True``, in which case, the backend will simulate a user-defined gate by computing its unitary matrix.

    The pulse backend does not allow simulation with user-defined gates.

An example demonstrating configuring options:

.. code-block::

    backend = QiskitCircuitSimulator()
    job = backend.run(circ, shots=3000)
    result = job.result()

We provided the value of shots explicitly, hence our options for the simulation are set as: ``shots=3000`` and ``allow_custom_gate=True``.

Another example:

.. code-block::
    
    backend = QiskitCircuitSimulator()
    job = backend.run(circ, shots=3000, allow_custom_gate=False)
    result = job.result()