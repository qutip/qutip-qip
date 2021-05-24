.. _introduction:

************
Introduction
************

The qutip-qip package
=====================

The qutip-qip package used to be a module ``qutip.qip`` under `QuTiP (Quantum Toolbox in Python) <http://qutip.org/index.html>`_.
From QuTiP 5.0, the community has decided to decrease the size of the core QuTiP package by reducing the external dependencies, in order to simplify maintenance.
Hence a few modules are separated from the core QuTiP and will become QuTiP family packages.
They are still maintained by the QuTiP team but hosted under different repositories in the `QuTiP organization <https://github.com/qutip>`_.

The qutip-qip package, QuTiP quantum information processing, aims at providing basic tools for quantum computing simulation both for simple quantum algorithm design and for experimental realization.
Compared to other libraries for quantum information processing, qutip-qip puts additional emphasis on the physics layer and the interaction with the QuTiP package.
It offers two different approaches for simulating quantum circuits, one with :class:`.QubitCircuit` calculating unitary evolution under quantum gates by matrix product, another called :class:`.Processor` using open system solvers in QuTiP to simulate noisy quantum device.
