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
The minimal Python version supported is Python 3.7.


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

Additional software for Plotting Circuits
=========================================
In order to plot circuits, the following non-Python packages are needed:

LaTeX
-----
The circuit plotting function in QuTiP uses LaTeX and a few LaTeX packages including ``qcircuit``, ``pdfcrop`` and ``braket``.

**For Linux and Mac** :
You need to install a TeX distribution such as `TeX Live <https://www.tug.org/texlive/>`_. You can either install it through ``apt-get``/``brew`` or using their installer on the website.
If you would prefer to avoid installing the full ``texlive`` distribution (which is a few Gigabyte large), `this link <https://tex.stackexchange.com/a/504566/203959>`_
has some useful discussion on selectively installing part of the packages. In particular:

* ``braket`` will be installed as a part of ``texlive-latex-extra``
* ``qcircuit`` will be installed as a part of ``texlive-pictures``
* ``pdfcrop`` and ``pdflatex`` are installed when a minimal ``texlive`` is installed.

.. note::
  Do not install ``pdfcrop`` via pip because the package's old dependencies will clash
  with newer dependencies of other programs installed in the ``qutip-qip`` virtual
  environment.

**For Windows** :
We recommend installing `MiKTeX <https://miktex.org/>`_, which will automatically install necessary packages like ``qcircuit`` for you when it is used. It will only take a few more minutes in your first attempt at plotting a circuit.
In addition, you also need to install perl for ``pdfcrop``.

ImageMagick and Ghostscript
---------------------------
In order to display the circuit in Jupyter notebook, we need to convert it to png
format. To do that, you will need to install `Ghostscript <https://www.ghostscript.com/doc/current/Make.htm>`_
and `ImageMagick <https://imagemagick.org/script/install-source.php>`_.
The first is responsible for reading the pdf file while the second will convert it to png.


.. note::
    You `might need to make changes <https://stackoverflow.com/a/52863413/10241324>`_ to ``policy.xml`` if the following error occurs :

    .. code-block:: text

        RuntimeError: convert-im6.q16: not authorized `qcirc.pdf' @ error/constitute.c/ReadImage/412.
        convert-im6.q16: no images defined `qcirc.png' @ error/convert.c/ConvertImageCommand/3258.


The output of a circuit plot will be output in a `Jupyter notebook <https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html>`_.
In your virtual environment, start `a Jupyter notebook server <https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server>`_ and run your examples
in a notebook. If you try to access a circuit plot in a terminal or IPython console,
you will only be able to access the location of this image in your device's memory.



pdf2svg
-------
To convert the circuit into svg format, you will need to install ``pdf2svg``.
Please visit `their website <https://github.com/dawbarton/pdf2sv>`_ for installation guide.

.. note::
    If you want to check whether all dependencies are installed,
    see if the following three commands work correctly:
    ``pdflatex``, ``pdfcrop`` and ``magick anypdf.pdf anypdf.png``,
    where ``anypdf.pdf`` is any pdf file you have.

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

.. code-block:: bash

    pytest tests

