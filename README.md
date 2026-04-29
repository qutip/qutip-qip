# qutip-qip

[![Tests](https://github.com/qutip/qutip-qip/actions/workflows/test.yaml/badge.svg?branch=master)](https://github.com/qutip/qutip-qip/actions/workflows/test.yaml)
[![Documentation Status](https://readthedocs.org/projects/qutip-qip/badge/?version=latest)](https://qutip-qip.readthedocs.io/en/latest/)
[![Coverage Status](https://coveralls.io/repos/github/qutip/qutip-qip/badge.svg)](https://coveralls.io/github/qutip/qutip-qip)
[![Maintainability](https://qlty.sh/gh/qutip/projects/qutip-qip/maintainability.svg)](https://qlty.sh/gh/qutip/projects/qutip-qip)
[![PyPI version](https://badge.fury.io/py/qutip-qip.svg)](https://badge.fury.io/py/qutip-qip)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)
[![DOI](https://img.shields.io/badge/DOI-10.22331%2Fq--2022--01--24--630-blue.svg)](https://doi.org/10.22331/q-2022-01-24-630)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

QuTiP-QIP is the QuTiP package for Quantum Information Processing. Compared to other libraries, QuTiP-QIP places stronger emphasis on the underlying physics and seamless integration with QuTiP.

The package offers two approaches for simulating quantum circuits:
- A standard matrix-based approach for unitary evolution under quantum gates.
- A Pulse level approach to simulate Quantum Circuits on noisy Quantum devices (e.g. Superconducting Qubits, Spin Chains) using QuTiP's open system solvers.


## Quick start

To install the package, use

```bash
pip install qutip-qip
```

You may refer to the [Installation guide](https://qutip-qip.readthedocs.io/en/latest/installation.html) for details on optional dependencies (feature-sets) available, installing `qutip-qip` from source etc.

```python
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import H, CX

qc = QubitCircuit(2)
qc.add_gate(H, 0)
qc.add_gate(CX, 0, 1)
state = qc.run()
print(state)

# You can optionally draw the circuit if you have matplotlib installed
qc.draw()
```

This simple example creates an entangled state known as a Bell state. We can run the above circuit on a Spin Chain Processor.

```python
from qutip import basis
from qutip_qip.device import LinearSpinChain

processor = LinearSpinChain(num_qubits=2)
processor.load_circuit(qc)
init_state = basis([2, 2], [0, 0]) # |00> state
result = processor.run_state(init_state)

final_state = result.states[-1]
print(final_state)
```
We can compare the fidelity between the Pulse level simulation (similar to how circuits run on actual QPU backends) vs state from perfect matrix product based simulation.

```python
from qutip import fidelity

print(fidelity(state, final_state))
```


## Documentation and Tutorials

Documentation and tutorials for `qutip-qip` can be found at [qutip-qip.readthedocs.io/](https://qutip-qip.readthedocs.io/en/stable/).

Code examples used in the publication *[Pulse-level noisy quantum circuits with QuTiP](https://quantum-journal.org/papers/q-2022-01-24-630)* updated for the latest version of `qutip-qip`, can be found in the [pulse-paper](https://github.com/qutip/qutip-qip/tree/master/pulse-paper) folder.


## Migrating from qutip.qip

The `qutip-qip` package was previously a module ``qutip.qip`` under [QuTiP (Quantum Toolbox in Python)](https://qutip.readthedocs.io/en/stable/). Starting from QuTiP 5.0, the community decided to decrease the size of the core QuTiP package, in order to simplify maintenance and for the sub-packages to evolve more quickly. If you were using qutip `4.x` and now want to try out the new features included in this package, you can simply install this package and replace all the `qutip.qip` in your import statement with `qutip_qip`. Everything should work smoothly as usual.

# Community

This project and everyone participating in it are governed by the [Code of Conduct](https://github.com/qutip/qutip-qip/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions & Discussions
You can ask questions or answer other users' questions on our [Discussion Forum](https://github.com/qutip/qutip-qip/discussions) on GitHub or in the [QuTiP Discussion group](https://groups.google.com/forum/#!forum/qutip). You may also suggest new features or improvements you would like to see in the project.

## Reporting Bugs & Issues
The list of existing issues can be found [here](https://github.com/qutip/qutip-qip/issues). If you encounter a bug during your workflow, please first search the existing issues to check if it has already been reported. You may also contribute by adding additional details or context to open issues.

If you do not find a related issue, please open a new [Issue](https://github.com/qutip/qutip-qip/issues/new) on GitHub. We will review and address it as soon as possible.

## Contributing
Contributions to `qutip-qip` are welcome. Please read [CONTRIBUTING.md](https://github.com/qutip/qutip-qip/blob/master/CONTRIBUTING.md) for details.

## Citing qutip-qip
If you use `qutip-qip` in your research, please cite this [paper](https://quantum-journal.org/papers/q-2022-01-24-630). The BibTeX file can be found [here](https://github.com/qutip/qutip-qip/tree/master/doc/source/qutip_qip.bib).

## Support

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)

This package is supported and maintained by the same developers group as QuTiP.

QuTiP development is supported by [Nori's lab](https://nori-physics.org/) at RIKEN, by the University of Sherbrooke, by Chalmers University of Technology, by Macquarie University and by Aberystwyth University, [among other supporting organizations](http://qutip.org/#supporting-organizations). We also thank the [Google Summer of Code Programme](https://summerofcode.withgoogle.com/) for supporting students to work on QuTiP and related projects over the years.
