*********
Changelog
*********


Version 0.4.1 (April 7, 2025)
+++++++++++++++++++++++++++++

This update includes improvements such as a new test for verifying that circuit files are correctly saved in PNG, TXT, and PDF formats, and the addition of a package entry point with about information. Bug fixes address a correction in the ZZ coefficients calculation in the superconducting qubits model and ensure that expand_operator outputs the expected data type.

Improvements
------------
- Add a test to verify that the Matplotlib, Text, and LaTeX renderers saves circuit files in PNG, TXT, and PDF formats. (`#260 <https://github.com/qutip/qutip-qip/pull/260>`_ by Rushiraj Gadhvi)
- Add family package entry point with about information. (`#261 <https://github.com/qutip/qutip-qip/pull/261>`_ by Simon Cross)

Bug Fixes
---------
- Use ``wq`` instead of ``wr`` for ``zz_coeff`` in ``SCqubits`` modelling. (`#270 <https://github.com/qutip/qutip-qip/pull/270>`_ by Ferris Prima Nugraha)
- Ensure the output of ``expand_operator`` is the expected dtype. (`#273 <https://github.com/qutip/qutip-qip/pull/273>`_ by Eric Giguère)


Version 0.4.0 (Nov 23, 2024)
++++++++++++++++++++++++++++

This release adds two new quantum circuit renderers -- one based on matplotlib and another that renders the circuit as text -- and improves the circuit simulation speed.

New features
------------
- Add matplotlib-based and text-based quantum circuit renderers. The new rendering framework is more flexible and can be used in different environments without latex dependencies. (`#239 <https://github.com/qutip/qutip-qip/pull/239>`_ and `#244 <https://github.com/qutip/qutip-qip/pull/244>`_ by Rushiraj Gadhvi)

Improvements
------------
- `einsum` is used to speed up the circuit simulation by up to 10 times. (`#225 <https://github.com/qutip/qutip-qip/pull/225>`_ by Boxi Li)


Miscellaneous
-------------
- The QIR support is removed. (`#238 <https://github.com/qutip/qutip-qip/pull/238>`_ by Boxi Li)

Version 0.3.2 (Sept 3, 2024)
++++++++++++++++++++++++++++

Improvements
------------
- Add RZX gate to the SCQubits model. Fix bugs on the ZX strength. (`#245 <https://github.com/qutip/qutip-qip/pull/245>`_)
- Compile the documentation using qutip v5 (`#246 <https://github.com/qutip/qutip-qip/pull/246>`_)

Bug Fixes
---------
- Add phi parameter to expand_operator call. (`#237 <https://github.com/qutip/qutip-qip/pull/237>`_ by Neill Lambert)
- Fix the warning on MacOS. (`#240 <https://github.com/qutip/qutip-qip/pull/240>`_)
- Fix a bug in :class:`~.QubitCircuit.add_1q_gate`. (`#250 <https://github.com/qutip/qutip-qip/pull/250>`_ issue raised by hatlabcz Chao Zhou)


Version 0.3.1 (April 21, 2024)
++++++++++++++++++++++++++++++

Improvements
------------
- Update qiskit support version to ``0.46.*`` (`#232 <https://github.com/qutip/qutip-qip/pull/232>`_)

Bug Fixes
---------
- Remove comments after meaningful statements in QASM reader. (`#218 <https://github.com/qutip/qutip-qip/pull/218>`_ by NoriyukiK-1qbit)
- Fix circuit plot error for gates with classical controls when ``reverse_states=False``. (`#221 <https://github.com/qutip/qutip-qip/pull/221>`_ by Kevin P. O'Brien)
- Update the RTD configuration file to execute ``create_tutorials_html.py`` (`#222 <https://github.com/qutip/qutip-qip/pull/222>`_, reported by Ramyashri Padmanabhakumar)
- Remove the unnecessary use of circuit.propagatos (`#226 <https://github.com/qutip/qutip-qip/pull/226>`_)
- Use CSR as default for ``expand_operator`` (`#227 <https://github.com/qutip/qutip-qip/pull/227>`_)
- Update the parsing mode for ``read_qasm`` (`#224 <https://github.com/qutip/qutip-qip/pull/224>`_ reported by kevinab107)


Version 0.3.0 (July 19, 2023)
+++++++++++++++++++++++++++++

This release adds two major features and one minor feature to the package.

New features
------------
- **MAJOR** Add Variational Quantum Algorithms in a new module :obj:`qutip_qip.vqa`. `Unitary fund microgrant project <https://unitary.fund/posts/vqa_in_qutip.html>`_ (`#123 <https://github.com/qutip/qutip-qip/pull/123>`_)
- **MAJOR** qutip-qip backends for qiskit :obj:`qutip_qip.qiskit`. `Google Summer of Code project 2022 <https://summerofcode.withgoogle.com/archive/2022/projects/jFfaK3Su>`_ (`#155 <https://github.com/qutip/qutip-qip/pull/155>`_, `#159 <https://github.com/qutip/qutip-qip/pull/159>`_)
- Add class representation of quantum gates. (`#147 <https://github.com/qutip/qutip-qip/pull/123>`_)

Documentation
-------------
- Add synced qutip-qip tutorials to documentation on Read the docs (`#207 <https://github.com/qutip/qutip-qip/pull/207>`_)

Version 0.2.3 (December 12, 2022)
+++++++++++++++++++++++++++++++++

Improvements
------------
- Add DRAG correction for the ``SCQubits`` simulator. Also improve the interpolation for the idling gates for continuous pulse shape. (`#156 <https://github.com/qutip/qutip-qip/pull/156>`_)

Bug Fixes
---------
- Fix usage of ``rand_unitary`` and ``rand_ket``. (`#163 <https://github.com/qutip/qutip-qip/pull/163>`_)
- Fix the final time calculation in ``GateCompiler.compile``. (`#162 <https://github.com/qutip/qutip-qip/pull/162>`_)
- Fix bug for circuit with selected measurement result. (`#167 <https://github.com/qutip/qutip-qip/pull/167>`_)
- Fix bugs in the analytical pulse shapes. (`#169 <https://github.com/qutip/qutip-qip/pull/169>`_)
- Fix the swap gate plotting. (`#170 <https://github.com/qutip/qutip-qip/pull/170>`_) and (`#177 <https://github.com/qutip/qutip-qip/pull/177>`_)
- Filter the measurement probability (`#182 <https://github.com/qutip/qutip-qip/pull/182>`_)
- Add a default max_step based on the circuit time (`#183 <https://github.com/qutip/qutip-qip/pull/183>`_)

Miscellaneous
-------------
- Fix parsing of package versions in packaging 22. (`#181 <https://github.com/qutip/qutip-qip/pull/181>`_)



Version 0.2.2 (June 20, 2022)
+++++++++++++++++++++++++++++

Improvements
------------
- Correct the spelling of Hadamard. (`#132 <https://github.com/qutip/qutip-qip/pull/132>`_)
- Support the `dev.major` branch of `qutip`. (`#136 <https://github.com/qutip/qutip-qip/pull/136>`_)
- Export circuit plots as png with dpi. (`#139 <https://github.com/qutip/qutip-qip/pull/139>`_)

Bug Fixes
---------
- Fix Circuit Plotting for non-reversed states. (`#137 <https://github.com/qutip/qutip-qip/pull/137>`_)
- Fix add_gate behavior when supplied multiple indices. (`#142 <https://github.com/qutip/qutip-qip/pull/142>`_)
- Quantum circuits inherit user gates from add_circuit. (`#144 <https://github.com/qutip/qutip-qip/pull/144>`_)


Version 0.2.1 (Feb 20, 2022)
++++++++++++++++++++++++++++

This release includes a revision of the documentation and adds more code examples in the API doc strings.

Bug Fixes
---------
- Remove the check on the initial state dimension in ``QubitCircuit.run()`` (`#127 <https://github.com/qutip/qutip-qip/pull/127>`_)

Improvements
------------
-  Rewrite the documentation for the pulse-level simulation. (`#121 <https://github.com/qutip/qutip-qip/pull/121>`_)
-  Add more code examples in the doc strings. (`#126 <https://github.com/qutip/qutip-qip/pull/126>`_)


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