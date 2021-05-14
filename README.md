# qutip-qip

[![build](https://github.com/qutip/qutip-qip/workflows/Tests/badge.svg)](https://github.com/qutip/qutip-qip/actions)
[![Documentation Status](https://readthedocs.org/projects/qutip-qip/badge/?version=latest)](https://qutip-qip.readthedocs.io/en/latest/)

The qutip-qip package used to be a module ``qutip.qip`` under [QuTiP (Quantum Toolbox in Python)](http://qutip.org/index.html).
From QuTiP 5.0, the community has decided to decrease the size of the core QuTiP package by reducing the external dependencies, in order to simplify maintenance.
Hence a few modules are separated from the core QuTiP and will become QuTiP family packages.
They are still maintained by the QuTiP team but hosted under different repositories in the [QuTiP organization](https://github.com/qutip).

The qutip-qip package, QuTiP quantum information processing, aims at providing basic tools for quantum computing simulation both for simple quantum algorithm design and for experimental realization.
Compared to other libraries for quantum information processing, qutip-qip puts additional emphasis on the physics layer and the interaction with the QuTiP package.
The package offers two different approaches for simulating quantum circuits, one with `QubitCircuit` calculating unitary evolution under quantum gates by matrix product, another called `Processor` using open system solvers in QuTiP to simulate noisy quantum device.

If you would like to know the future development plan and ideas, have a look at the [discussion panel](https://github.com/qutip/qutip-qip/discussions) as well as the [qutip documentation for ideas](https://github.com/qutip/qutip-doc/tree/master/development/ideas).

Installation
------------
To install the package, download to source code and run
```
pip install qutip_qip
```

If you want to edit the source code, please download the source code and run the following command under the folder with `setup.cfg`
```
pip install -e .
```

To build and test the documentation, additional packages need to be installed:

```
pip install matplotlib sphinx numpydoc sphinx_rtd_theme
```

Under the `docs` directory, use
```
make html
```
to build the documentation, or
```
make doctest
```
to test the code in the documentation.

Documentation
-------------

The documentation of `qutip-qip` updated to the latest development version is hosted at [qutip-qip.readthedocs.io/](https://qutip-qip.readthedocs.io/en/latest/).

Testing
------------
To test the installation from a download of the source code, run from the `qutip-qip` directory
```
    pytest tests
```

Support
-------
This package is supported and maintained by the same developers group as QuTiP.

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)


QuTiP development is supported by [Nori's lab](http://dml.riken.jp/)
at RIKEN, by the University of Sherbrooke, by Chalmers University of Technology, by Macquarie University and by Aberystwyth University,
[among other supporting organizations](http://qutip.org/#supporting-organizations).

License
-------
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE.txt file are satisfied.
