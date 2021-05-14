************
Installation
************

.. _prerequisites:

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

    sphinx numpydoc sphinx_rtd_theme

are used to build and test the documentation.

A few other packages such as LaTeX is used for circuit plotting, please refer to the main documentation section for detailed instruction.

.. _installation:

Install qutip-qip from source code
==================================

To install the package, download to source code from `GitHub website <https://github.com/qutip/qutip-qip>`_ and run

.. code-block:: bash

    pip install .

under the directory containing the ``setup.py`` file.

If you want to edit the code, use instead

.. code-block:: bash

    pip install -e .
