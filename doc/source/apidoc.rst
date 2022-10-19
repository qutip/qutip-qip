qutip\_qip package
==================


.. toctree::
   :maxdepth: 1


Gate-level simulation
-------------------------
Simulation based on operator-state multiplication.

.. autosummary::
   :toctree: apidoc/
   :template: autosummary/module.rst

   qutip_qip.circuit
   qutip_qip.operations
   qutip_qip.qubits
   qutip_qip.decompose
   qutip_qip.qasm
   qutip_qip.qir
   qutip_qip.vqa

Pulse-level simulation
----------------------
Simulation based on the master equation.

.. autosummary::
   :toctree: apidoc/
   :template: autosummary/module.rst

   qutip_qip.device
   qutip_qip.compiler
   qutip_qip.pulse
   qutip_qip.noise

Qiskit Circuit Simulation
--------------------------
Simulation of qiskit circuits based on qutip_qip backends.

.. autosummary::
   :toctree: apidoc/
   :template: autosummary/module.rst

   qutip_qip.qiskit