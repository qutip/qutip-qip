.. _qip_qiskit:

**********************************
`qutip-qip` as a Qiskit backend
**********************************

This submodule was implemented by `Shreyas Pradhan <shpradhan12@gmail.com>`_ as part of Google Summer of Code 2022.

Overview
===============

This submodule provides an interface to simulate circuits made in qiskit.

Gate-level simulation on qiskit circuits is possible with :class:`.QiskitCircuitSimulator`. Pulse-level simulation is possible with :class:`.QiskitPulseSimulator` which supports simulation using the :class:`.LinearSpinChain`, :class:`.CircularSpinChain` and :class:`.DispersiveCavityQED` pulse processors.

Running a qiskit circuit with `qutip_qip`
==========================================

After constructing a circuit in qiskit, either of the qutip_qip based backends (:class:`.QiskitCircuitSimulator` and :class:`.QiskitPulseSimulator`) can be used to run that circuit.

Example
--------

Let's try constructing and simulating a qiskit circuit.

We define a simple circuit as follows:


.. plot::
    :context: close-figs

    .. doctest:: 
        :hide:

        >>> import random
        >>> random.seed(1)

    .. doctest::
        :options: +SKIP

        >>> from qiskit import QuantumCircuit
        >>> circ = QuantumCircuit(2,2)
    
        >>> circ.h(0) 
        >>> circ.h(1)
        >>> circ.measure(0,0)
        >>> circ.measure(1,1)

    Let's run this on the :class:`.QiskitCircuitSimulator` backend:

    .. doctest::
        :options: +SKIP

        >>> from qutip_qip.qiskit import QiskitCircuitSimulator
        >>> backend = QiskitCircuitSimulator()
        >>> job = backend.run(circ)
        >>> result = job.result()
    
    The result object inherits from the :class:`qiskit.result.Result` class. Hence, we can use it's functions like ``result.get_counts()`` as required. We can also access the final state with ``result.data()['statevector']``.
    
    .. code-block::

        >>> result.data()['statevector']
        Statevector([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j], dims=(2, 2))

    .. doctest::
        :options: +SKIP

        >>> from qiskit.visualization import plot_histogram
        >>> plot_histogram(result.get_counts())

.. plot:: 
    :context: close-figs

    Now, let's run the same circuit on :class:`.QiskitPulseSimulator`.

    While using a pulse processor, we define the circuit without measurements.
    
    .. note::
    
        The pulse-level simulator does not support measurement. Please use :obj:`qutip.measure` to process the result manually. By default, all the qubits will be measured at the end of the circuit.

    .. _pulse circ:

    .. doctest::
        :options: +SKIP

        >>> pulse_circ = QuantumCircuit(2,2)
        >>> pulse_circ.h(0)
        >>> pulse_circ.h(1)

    To use the :class:`.QiskitPulseSimulator` backend, we need to define the processor on which we want to run the circuit. This includes defining the pulse processor model with all the required parameters including noise. 

    Different hardware parameters can be supplied here for :obj:`.LinearSpinChain`. Please refer to the documentation for details.
    
    .. doctest::
        :options: +SKIP

        >>> from qutip_qip.device import LinearSpinChain
        >>> processor = LinearSpinChain(num_qubits=2)

    Now that we defined our processor (:class:`.LinearSpinChain` in this case), we can use it to perform the simulation: 

    .. doctest::
        :options: +SKIP

        >>> from qutip_qip.qiskit import QiskitPulseSimulator

        >>> pulse_backend = QiskitPulseSimulator(processor)
        >>> pulse_job = pulse_backend.run(pulse_circ)
        >>> pulse_result = pulse_job.result()

    .. _pulse plot:

    .. doctest::
        :options: +SKIP

        >>> plot_histogram(pulse_result.get_counts())


Configurable Options
========================

Qiskit's interface allows us to provide some options like ``shots`` while running a circuit on a backend. We also have provided some options for the qutip_qip backends.

``shots``
-------------
``shots`` is the number of times measurements are sampled from the simulation result. By default it is set to ``1024``.

``allow_custom_gate``
-----------------------
``allow_custom_gate``, when set to ``False``, does not allowing simulating circuits that have user-defined gates; it will throw an error in that case. By default, it is set to ``True``, in which case, the backend will simulate a user-defined gate by computing its unitary matrix.

.. note::
    
    Although you can pass this option while running a circuit on pulse backends, you need to make sure that the gate is supported by the backend simulator :obj:`.Processor` in ``qutip-qip``.

An example demonstrating configuring options:

.. doctest::

    backend = QiskitCircuitSimulator()
    job = backend.run(circ, shots=3000)
    result = job.result()

We provided the value of shots explicitly, hence our options for the simulation are set as: ``shots=3000`` and ``allow_custom_gate=True``.

Another example:

.. doctest::
    
    backend = QiskitCircuitSimulator()
    job = backend.run(circ, shots=3000, allow_custom_gate=False)
    result = job.result()


Noise
=======

Real quantum devices are not ideal and are bound to have some amount of noise in them. One of the uses of having the pulse backends is the ability to add noise to our device.

Let's look at an example where we add some noise to our circuit and see what kind of bias it has on the results. We'll use the same circuit we used :ref:`above<pulse circ>`.

Let's use the :class:`.CircularSpinChain` processor this time with some noise.

.. plot:: 
    :context: close-figs

    .. doctest::
        :options: +SKIP
        
        >>> from qutip_qip.device import CircularSpinChain
        >>> processor = CircularSpinChain(num_qubits=2, t1=0.3)

    If we ran this on a processor without noise we would expect all states to be approximately equiprobable, like we saw :ref:`above<pulse plot>`.

    .. doctest::
        :options: +SKIP

        >>> noisy_backend = QiskitPulseSimulator(processor)
        >>> noisy_job = noisy_backend.run(pulse_circ)
        >>> noisy_result = noisy_job.result()
    
    ``t1=0.3`` will cause amplitude damping on all qubits, and hence, ``0`` is more probable than ``1`` in the final output for all qubits.

    We can see what the result looks like in the density matrix format:

    .. code-block::

        >>> noisy_result.data()['statevector']
        DensityMatrix([[ 0.4484772 +0.00000000e+00j,  0.04130281+2.46325222e-01j,
                 0.04130281+2.46325222e-01j, -0.13148987+4.53709696e-02j],
               [ 0.04130281-2.46325222e-01j,  0.22120721+0.00000000e+00j,
                 0.13909747-1.10349672e-17j,  0.02037223+1.21497634e-01j],
               [ 0.04130281-2.46325222e-01j,  0.13909747+1.10349672e-17j,
                 0.22120721+0.00000000e+00j,  0.02037223+1.21497634e-01j],
               [-0.13148987-4.53709696e-02j,  0.02037223-1.21497634e-01j,
                 0.02037223-1.21497634e-01j,  0.10910838+0.00000000e+00j]],
              dims=(2, 2))

    .. doctest::
        :options: +SKIP

        >>> plot_histogram(noisy_result.get_counts())
    
    