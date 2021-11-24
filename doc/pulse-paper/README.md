This folder contains code examples used in the publication [*Pulse-level noisy quantum circuits with QuTiP*](https://arxiv.org/abs/2105.09902). To run the examples, please first install the software package qutip-qip
```
pip install qutip_qip[full] joblib
```
All examples are self-contained and running the code should reproduce the plots used in the paper.

The following table summarizes the sections in the paper and the corresponding code examples:

| Section | Code example |
| ----------- | ----------- |
| Section 4 | `main_example.py`|
| Appendix A | `dj_algorithm.py` |
| Fig.4 and Appendix C | `customize.py` |
| Fig.5 | `decoherence.py` |
| Section 5 | `deutsch_jozsa.qasm` and `deutsch_jozsa-qasm.py` |
| Appendix B | `qft.py` |