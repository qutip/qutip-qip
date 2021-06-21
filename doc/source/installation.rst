************
Installation
************

.. _quickstart:

Quick start
===========
To install the package ``qutip-qip`` from PyPI, use

.. code-block:: bash

    pip install qutip-qip

Migrating from ``qutip.qip``
============================
As the :ref:`introduction` suggested, this package is based on a module in the `QuTiP <http://qutip.org/docs/latest/>`_ package ``qutip.qip``.
If you were using the ``qutip`` package and now want to try out the new features included in this package, you can simply install this package and replace all the ``qutip.qip`` in your import statement with ``qutip_qip``. Everything should work smoothly as usual.

.. _prerequisites:

Prerequisites
=============
This package is built upon QuTiP, of which the installation guide can be found at on `QuTiP Installation <http://qutip.org/docs/latest/installation.html>`_.
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

Plotting Circuits
------------------
In order to plot circuits, following packages are needed.

``texlive`` is one of the packages that could be installed for `LaTeX`.

.. code-block::

  apt-get texlive texlive-latex-base texlive-latex-recommended texlive-latex-extra


``ImageMagick`` is needed to display the circuit diagrams. The installation
instructions are available `here <https://github.com/conda-forge/imagemagick-feedstock#installing-imagemagick>`_.

You `might need <RuntimeError: convert-im6.q16: not authorized `qcirc.pdf' @ error/constitute.c/ReadImage/412.
convert-im6.q16: no images defined `qcirc.png' @ error/convert.c/ConvertImageCommand/3258.>`_ to make changes to ``policy.xml`` if following error occurs :

.. code-block::
  RuntimeError: convert-im6.q16: not authorized `qcirc.pdf' @ error/constitute.c/ReadImage/412.
  convert-im6.q16: no images defined `qcirc.png' @ error/convert.c/ConvertImageCommand/3258.

.. _installation:

Install qutip-qip from source code
==================================

To install the package, download to source code from `GitHub website <https://github.com/qutip/qutip-qip>`_ and run

.. code-block:: bash

    pip install .

under the directory containing the ``setup.cfg`` file.

If you want to edit the code, use instead

.. code-block:: bash

    pip install -e .

To test the installation from a download of the source code, run from the `qutip-qip` directory
```
pytest tests
```
