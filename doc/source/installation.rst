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

.. _circuit_plot_packages:

Plotting Circuits
------------------
In order to plot circuits, following packages are needed - ``LaTex``, ``ImageMagick``,
``QCircuit``, ``pdfcrop`` and ``pdflatex``.

**For Linux** : ``LaTex`` is needed to plot the circuits. If you would prefer to avoid installing the full ``texlive``
package for ``LaTex``, `this link has <https://tex.stackexchange.com/a/504566/203959>`_
some useful discussion on selectively installing smaller packages.

* ``braket`` will be installed as a part of ``texlive-latex-extra``
* ``qcircuit`` will be installed as a part of ``texlive-pictures``
* ``pdfcrop`` and ``pdflatex`` are installed when a minimal ``texlive`` is installed.

.. note::
  Do not install ``pdfcrop`` via pip because the package's old dependencies will clash
  with newer dependencies of other programs installed in the ``qutip-qip`` virtual
  environment.

``ImageMagick`` can be installed as a conda package via `this link <https://github.com/conda-forge/imagemagick-feedstock#installing-imagemagick>`_
or use this `link <https://imagemagick.org/script/download.php>`_ to install
via source. This package along with ``pdfcrop`` and ``pdflatex`` are
needed to display the circuit diagrams.


You `might need to make changes <https://stackoverflow.com/a/52863413/10241324>`_ to ``policy.xml`` if following error occurs :

.. code-block:: text

  RuntimeError: convert-im6.q16: not authorized `qcirc.pdf' @ error/constitute.c/ReadImage/412.
  convert-im6.q16: no images defined `qcirc.png' @ error/convert.c/ConvertImageCommand/3258.

The output of a circuit plot will be output in a `Jupyter notebook <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`_.
In your virtual environment, start a Jupyter notebook and run your examples
in a notebook. If you try to access a circuit plot in a terminal or IPython console,
you will be able to access the location of this image in memory.


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
