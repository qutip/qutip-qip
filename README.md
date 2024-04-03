# qutip-qip

[![build](https://github.com/qutip/qutip-qip/workflows/Tests/badge.svg)](https://github.com/qutip/qutip-qip/actions)
[![Documentation Status](https://readthedocs.org/projects/qutip-qip/badge/?version=stable)](https://qutip-qip.readthedocs.io/en/stable/)
[![PyPI version](https://badge.fury.io/py/qutip-qip.svg)](https://badge.fury.io/py/qutip-qip)
[![arXiv paper](https://img.shields.io/badge/arXiv-2105.09902-<COLOR>.svg)](https://arxiv.org/abs/2105.09902)
[![Maintainability](https://api.codeclimate.com/v1/badges/30293d7b8eb249f8d679/maintainability)](https://codeclimate.com/github/qutip/qutip-qip/maintainability)
[![Coverage Status](https://coveralls.io/repos/github/qutip/qutip-qip/badge.svg)](https://coveralls.io/github/qutip/qutip-qip)

The qutip-qip package used to be a module ``qutip.qip`` under [QuTiP (Quantum Toolbox in Python)](http://qutip.org/index.html).
From QuTiP 5.0, the community has decided to decrease the size of the core QuTiP package by reducing the external dependencies, in order to simplify maintenance.
Hence a few modules are separated from the core QuTiP and will become QuTiP family packages.
They are still maintained by the QuTiP team but hosted under different repositories in the [QuTiP organization](https://github.com/qutip).

The qutip-qip package, QuTiP quantum information processing, aims at providing basic tools for quantum computing simulation both for simple quantum algorithm design and for experimental realization.
Compared to other libraries for quantum information processing, qutip-qip puts additional emphasis on the physics layer and the interaction with the QuTiP package.
The package offers two different approaches for simulating quantum circuits, one with `QubitCircuit` calculating unitary evolution under quantum gates by matrix product, another called `Processor` using open system solvers in QuTiP to simulate noisy quantum device.

If you would like to know the future development plan and ideas, have a look at the [discussion panel](https://github.com/qutip/qutip-qip/discussions) as well as the [qutip documentation for ideas](https://qutip.readthedocs.io/en/stable/development/ideas.html).

Quick start
-----------
To install the package, use
```
pip install qutip-qip
```

Migrating from ``qutip.qip``
--------------------------
As the introduction suggested, this package is based on a module in the [QuTiP](http://qutip.org/docs/latest/) package `qutip.qip`.
If you were using the `qutip` package and now want to try out the new features included in this package, you can simply install this package and replace all the `qutip.qip` in your import statement with `qutip_qip`. Everything should work smoothly as usual.

Documentation and tutorials
-------------

The documentation of `qutip-qip` updated to the latest development version is hosted at [qutip-qip.readthedocs.io/](https://qutip-qip.readthedocs.io/en/stable/).
Tutorials related to using quantum gates and circuits in `qutip-qip` can be found [*here*](https://qutip.org/tutorials#quantum-information-processing) and those related to using noise simulators areavailable at [*this link*](https://qutip.org/tutorials#nisq). 

Code examples used in the article [*Pulse-level noisy quantum circuits with QuTiP*](https://quantum-journal.org/papers/q-2022-01-24-630), updated for the latest code version, are hosted in [this folder](https://github.com/qutip/qutip-qip/tree/master/doc/pulse-paper).

Installation from source
------------------------
If you want to edit the source code, please download the source code and run the following command under the root `qutip-qip` folder,
```
pip install --upgrade pip
pip install -e .
```
which makes sure that you are up to date with the latest `pip` version. Contribution guidelines are available [*here*](https://qutip-qip.readthedocs.io/en/latest/contribution-code.html). 

To build and test the documentation, additional packages need to be installed:

```
pip install pytest matplotlib sphinx numpydoc sphinx_rtd_theme
```

Under the `doc` directory, use
```
make html
```
to build the documentation, or
```
make doctest
```
to test the code in the documentation.

Testing
------------
To test the installation, choose the correct branch that matches with the version, e.g., `qutip-qip-0.2.X` for version 0.2. Then download the source code and run from the `qutip-qip` directory
```
pytest tests
```

Citing `qutip-qip`
------------
If you use `qutip-qip` in your research, please cite the [article](https://quantum-journal.org/papers/q-2022-01-24-630) as

```bibtex
@article{Li2022pulselevelnoisy,
  doi = {10.22331/q-2022-01-24-630},
  url = {https://doi.org/10.22331/q-2022-01-24-630},
  title = {Pulse-level noisy quantum circuits with {Q}u{T}i{P}},
  author = {Li, Boxi and Ahmed, Shahnawaz and Saraogi, Sidhant and Lambert, Neill and Nori, Franco and Pitchford, Alexander and Shammah, Nathan},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {6},
  pages = {630},
  month = jan,
  year = {2022}
}
```
Support
-------
This package is supported and maintained by the same developers group as QuTiP.

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)


QuTiP development is supported by [Nori's lab](http://dml.riken.jp/)
at RIKEN, by the University of Sherbrooke, by Chalmers University of Technology, by Macquarie University and by Aberystwyth University,
[among other supporting organizations](http://qutip.org/#supporting-organizations).

License
-------
[![license](https://img.shields.io/badge/license-New%20BSD-blue.svg)](http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE.txt file are satisfied.
