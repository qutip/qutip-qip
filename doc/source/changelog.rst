.. _changelog:

**********
Change Log
**********

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
