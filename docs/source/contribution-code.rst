.. _contribute_code:

*******************************
Contributing to the source code
*******************************

Build up an development environment
===================================

Please follow the instruction on the `QuTiP contribution guide <https://qutip.org/docs/latest/development/contributing.html#building>`_ to 
build a conda environment.
You don't need to build `qutip` in the editable mode unless you also want to contribute to `qutip`.
Instead, you need to install qutip-qip by downloading the source code and run

.. code-block:: bash

    pip install -e .

Docstrings for the code
=======================

Each class and function should be accompanied with a docstring
explaining the functionality, including input parameters and returned values.
The docstring should follow
`NumPy Style Python Docstrings <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_.

