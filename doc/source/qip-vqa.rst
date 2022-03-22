.. _qip_vqa:

******************************
Variational Quantum Algorithms
******************************

Variational Quantum Algorithms (VQAs) are represented by a parameterized quantum circuit, and include methods for defining a cost function for the circuit, and finding parameters that minimize this cost.


Constructing a VQA circuit
==========================

The :class:`.VQA` class allows for the construction of a parameterized circuit from :class:`.VQABlock` instances, which act as the gates of the circuit. In the most basic instance, a :class:`.VQA` should have:

====================  =================================================
Property                           Description
====================  =================================================
``n_qubits``          Positive integer number of qubits for the circuit 
``n_layers``          Positive integer number of repetitions of the 
                      layered elements of the circuit
``cost_method``       String referring to the method used to
                      evaluate the circuit's cost.

                      Either "OBSERVABLE", "BITSTRING", or "STATE".
====================  =================================================

For example:

.. code-block::

    from qutip_qip.vqa import VQA

    VQA_circuit = VQA(
                n_qubits=1,
                n_layers=1,
                cost_method="OBSERVABLE",
            )

After constructing this instance, we are ready begin adding elements to our parameterized circuit. Circuit elements in this module are represented by :class:`.VQABlock` instances. Fundamentally, the role of this class is to generate an operator for the circuit. To do this, it can store things such a number of free parameters, a Hamiltonian to exponentiate, or a string referring to a gate that has already been defined.

In the absence of specification, a VQA block will assume any :class:`~.Qobj` it is given to be a Hamiltonian, :math:`H`, and will generate a unitary with free parameter, :math:`\gamma`, as :math:`U(\gamma) = e^{-i \gamma H}`. For example, 

.. testcode::

    from qutip_qip.vqa import VQABlock
    from qutip import tensor, sigmax

    X_block = VQABlock(
      sigmax(), name="X Gate"
    )


We can add this block to our ``VQA_circuit`` with the :meth:`.VQA.add_block` method. Calling the :meth:`.VQA.export_image` method will allow us to see the effects of adding this block.

.. testcode::
   
    VQA_circuit.add_block(X_block)

    VQA_circuit.export_image("circ.png")

**Output**:

.. image:: /figures/vqa_circuit_with_x.png
