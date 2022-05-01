*********
Changelog
*********

Version 0.2.1 (Feb 20, 2022)
++++++++++++++++++++++++++++

This release includes a revision of the documentation and adds more code examples in the API doc strings.

Bug Fixes
---------
- Remove the check on the initial state dimension in ``QubitCircuit.run()`` (#127 <https://github.com/qutip/qutip-qip/pull/127>`_)

Improvements
------------
-  Rewrite the documentation for the pulse-level simulation. (#121 <https://github.com/qutip/qutip-qip/pull/121>`_)
-  Add more code examples in the doc strings. (#126 <https://github.com/qutip/qutip-qip/pull/126>`_)


Version 0.2.0 (Nov 26, 2021)
++++++++++++++++++++++++++++
This release adds a few new features to the pulse-level simulator.

PRs are collected `https://github.com/qutip/qutip-qip/milestone/3?closed=1 <https://github.com/qutip/qutip-qip/milestone/3?closed=1>`_.

Improvements
------------
- **MAJOR** Add the :obj:`.Model` class that represents the physical model including hardware parameters, control and drift Hamiltonians and noise objects. (`#105 <https://github.com/qutip/qutip-qip/pull/105>`_)
- Separate the gate definition from the QubitCircuit.propagators method (`#83 <https://github.com/qutip/qutip-qip/pull/83>`_)
- Support different pulse shapes. (`#85 <https://github.com/qutip/qutip-qip/pull/85>`_)
- Use autosummary to generate a summary of API docs. (`#103 <https://github.com/qutip/qutip-qip/pull/103>`_)
- Improve the scheduling algorithm. (`#105 <https://github.com/qutip/qutip-qip/pull/105>`_)

.. note:: 
    Compatibility Note: The behaviour of ``Processor.pulses`` changes significantly from version 0.1 to version 0.2. In 0.1, if no control coefficients are added, `pulses` contains a list of partially initialized :obj:`.Pulse` objects. They include control Hamiltonians but have no coefficients or tlist. This behaviour has changed. From 0.2, the list only includes controls that have non-trivial dynamics. To inspect the available control Hamiltonians, please use :obj:`.Processor.get_control` and :obj:`.Processor.get_control_labels`.


Version 0.1.2 (Nov 25, 2021)
++++++++++++++++++++++++++++
This micro release adds more thorough documentation for the project and fixes a few bugs in :obj:`.QubitCircuit` and :obj:`.Processor`.

PRs are collected at `https://github.com/qutip/qutip-qip/milestone/4?closed=1 <https://github.com/qutip/qutip-qip/milestone/4?closed=1>`_.

Improvements
------------
- Efficient Hadamard transform. (`#103 <https://github.com/qutip/qutip-qip/pull/103>`_)
- Make circuit latex code accessible in `QubitCircuit`. (`#108 <https://github.com/qutip/qutip-qip/pull/108>`_)


Bug Fixes
----------
- Fix the leaking noise objects in `Processor`. (`#89 <https://github.com/qutip/qutip-qip/pull/89>`_)
- Fix a bug in time-dependent collapse operators in  `Processor`. (`#107 <https://github.com/qutip/qutip-qip/pull/107>`_)


Version 0.1.1 (July 28, 2021)
+++++++++++++++++++++++++++++

This micro release adds more thorough documentation for the project and fixes a few bugs in :obj:`.QubitCircuit` and :obj:`.Processor`.

PRs are collected `here <https://github.com/qutip/qutip-qip/milestone/2?closed=1>`_.

Improvements
------------
- Improve the documentation.
- Workflows for releases and automatically building the documentation, migrated from ``qutip``. (`#49 <https://github.com/qutip/qutip-qip/pull/49>`_, `#78 <https://github.com/qutip/qutip-qip/pull/78>`_)
- The part of tex code taken from circuit is removed due to licence issue. Instead, the latex code now requires the user to install `qcircuit` in advance. (`#61 <https://github.com/qutip/qutip-qip/pull/61>`_)
- Rename :obj:`.Noise.get_noisy_dynamics` with :obj:`.Noise.get_noisy_pulses`. The new name is more appropriate because it returns a list of :obj:`.Pulse`, not a ``QobjEvo``. The old API is deprecated. (`#76 <https://github.com/qutip/qutip-qip/pull/76>`_)
- Add more thorough documentation for installing external dependencies for circuit plotting. (`#65 <https://github.com/qutip/qutip-qip/pull/65>`_)

Bug Fixes
---------
- Add the missing drift Hamiltonian to the method :obj:`.Processor.run_analytically`. It was missing because only the control part of the Hamiltonian is added. (`#74 <https://github.com/qutip/qutip-qip/pull/74>`_)
- Fix a few bugs in :obj:`.QubitCircuit`: Make `QubitCircuit.propagators_no_expand` private. It will be removed and replaced by :obj:`.QubitCircuit.propagators`. The attributes :obj:`.QubitCircuit.U_list` is also removed. (`#66 <https://github.com/qutip/qutip-qip/pull/66>`_)

Developer Changes
-----------------
- Documentation is moved from ``/docs`` to ``/doc``. (`#49 <https://github.com/qutip/qutip-qip/pull/49>`_, `#78 <https://github.com/qutip/qutip-qip/pull/78>`_)


Version 0.1.0 (May 14, 2021)
++++++++++++++++++++++++++++

This is the first release of qutip-qip, the Quantum Information Processing package in QuTiP.

The qutip-qip package used to be a module ``qutip.qip`` under `QuTiP (Quantum Toolbox in Python) <http://qutip.org/index.html>`_. From QuTiP 5.0, the community has decided to decrease the size of the core QuTiP package by reducing the external dependencies, in order to simplify maintenance. Hence a few modules are separated from the core QuTiP and will become QuTiP family packages. They are still maintained by the QuTiP team but hosted under different repositories in the `QuTiP organization <https://github.com/qutip>`_.

The qutip-qip package, QuTiP quantum information processing, aims at providing basic tools for quantum computing simulation both for simple quantum algorithm design and for experimental realization. Compared to other libraries for quantum information processing, qutip-qip puts additional emphasis on the physics layer and the interaction with the QuTiP package. The package offers two different approaches for simulating quantum circuits, one with :obj:`.QubitCircuit` calculating unitary evolution under quantum gates by matrix product, another called :obj:`.Processor` using open system solvers in QuTiP to simulate the execution of quantum circuits on a noisy quantum device.
