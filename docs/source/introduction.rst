.. _install:

************
Introduction
************

The qutip-qip package
=====================

The qutip-qip package used to be a module ``qutip.qip`` under `QuTiP (Quantum Toolbox in Python) <http://qutip.org/index.html>`_.
From QuTiP 5.0, the community has decided to reduce the size of the core QuTiP package for reducing the external dependency and ease of maintenance.
Hence a few modules are separated from the core QuTiP and become QuTiP family packages.
They are still maintained by the QuTiP team but hosted under different repositories in the `QuTiP organization <https://github.com/qutip>`_.

The qutip-qip package, QuTiP quantum information processing, aims at providing basic tools for quantum computing simulation both for simple quantum algorithm design and for experimental realization.
Compared to other libraries for quantum information processing, qutip-qip put additional emphasis on the physics layer and the interaction with the QuTiP package.
It offers two different approaches for simulating quantum circuits, one with :class:`~qutip_qip.QubitCircuit` calculating unitary evolution under quantum gates by matrix product, another called :class:`~qutip_qip.device.Processor` using open system solvers in QuTiP to simulate noisy quantum device.

.. _prerequisites

Prerequisites
=============
As the name indicates, this package is built upon QuTiP, the installation guide can be found at on `QuTiP Installation <http://qutip.org/docs/latest/installation.html>`_.
The only difference is that C++ compilers are not required here
since there is no run-time compiling for qutip-qip.
The minimal Python version supported is Python 3.6.


In particular, following packages are necessary for running qutip-qip

.. code-block:: bash

    numpy scipy cython qutip

The following to packages are used for plotting and testing:

.. code-block:: bash

    matplotlib pytest

In addition

.. code-block:: bash

    sphinx sphinx_rtd_theme doctest

are used to build the documentation.

A few other packages such as LaTeX is used for circuit plotting, please refer to the main documentation section for detailed instruction.

.. _installation

Installation
============

To install the package, download to source code from `GitHub website <https://github.com/qutip/qutip-qip>`_ and run

.. code-block:: bash

    pip install .

under the directory containing the ``setup.py`` file.

If you want to edit the code, use instead

.. code-block:: bash

    pip install -e .
