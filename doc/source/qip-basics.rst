.. _qip_intro:

******************************
Quantum Information Processing
******************************


.. _quantum_circuits:

Quantum Circuit
===============

The most common model for quantum computing is the quantum circuit model.
In QuTiP, we use :class:`.QubitCircuit` to represent a quantum circuit.
The circuit is characterized by registers and gates:

- **Registers**: The argument ``N`` specifies the number of qubit registers in the circuit
  and the argument ``num_cbits`` (optional) specifies the number of classical bits available for measurement
  and control.

- **Gates**: Each quantum gate is saved as a class object :class:`.Gate`
  with information such as gate name, target qubits and arguments.
  Gates can also be controlled on a classical bit by specifying the register number
  with the argument ``classical_controls``.

- **Measurements**: We can also carry out measurements on individual qubit (both in the middle and at the end of the circuit).
  Each measurement is saved as a class object :class:`.Measurement` with parameters such as `targets`,
  the target qubit on which the measurement will be carried out, and `classical_store`,
  the index of the classical register which stores the result of the measurement.

A circuit with the various gates and registers available is demonstrated below:

.. testcode::

  from qutip_qip.circuit import QubitCircuit
  from qutip_qip.operations import Gate
  from qutip import tensor, basis

  qc = QubitCircuit(N=2, num_cbits=1)
  swap_gate = Gate(name="SWAP", targets=[0, 1])

  qc.add_gate(swap_gate)
  qc.add_measurement("M0", targets=[1], classical_store=0) # measurement gate
  qc.add_gate("CNOT", controls=0, targets=1)
  qc.add_gate("X", targets=0, classical_controls=[0]) # classically controlled gate
  qc.add_gate(swap_gate)

  print(qc.gates)

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  [Gate(SWAP, targets=[0, 1], controls=None, classical controls=None, control_value=None),
   Measurement(M0, target=[1], classical_store=0),
   Gate(CNOT, targets=[1], controls=[0], classical controls=None, control_value=None),
   Gate(X, targets=[0], controls=None, classical controls=[0], control_value=None),
   Gate(SWAP, targets=[0, 1], controls=None, classical controls=None, control_value=None)]

Unitaries
=========

There are a few useful functions associated with the circuit object. For example,
the :meth:`.QubitCircuit.propagators` method returns a list of the unitaries associated
with the sequence of gates in the circuit. By default, the unitaries are expanded to the
full dimension of the circuit:

.. testcode::

  U_list = qc.propagators()
  print(U_list)

**Output**:

.. testoutput::

  [Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]
   [0. 0. 1. 0.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[0. 0. 1. 0.]
   [0. 0. 0. 1.]
   [1. 0. 0. 0.]
   [0. 1. 0. 0.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]]]

Another option is to only return the unitaries in their original dimension. This
can be achieved with the argument ``expand=False`` specified to the
:meth:`.QubitCircuit.propagators`.

.. testcode::

  U_list = qc.propagators(expand=False)
  print(U_list)

**Output**:

.. testoutput::

  [Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]
   [0. 0. 1. 0.]], Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
  Qobj data =
  [[0. 1.]
   [1. 0.]], Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
  Qobj data =
  [[1. 0. 0. 0.]
   [0. 0. 1. 0.]
   [0. 1. 0. 0.]
   [0. 0. 0. 1.]]]

.. _quantum_gates:

Gates
=====

The pre-defined gates for the class :class:`.Gate` are shown in the table below:

====================  ========================================
Gate name                           Description
====================  ========================================
"RX"                  Rotation around x axis
"RY"                  Rotation around y axis
"RZ"                  Rotation around z axis
"X"                   Pauli-X gate
"Y"                   Pauli-Y gate
"Z"                   Pauli-Z gate
"S"                   Single-qubit rotation or Z90
"T"                   Square root of S gate
"SQRTNOT"             Square root of NOT gate
"SNOT"                Hardmard gate
"PHASEGATE"           Add a phase one the state 1
"CRX"                 Controlled rotation around x axis
"CRY"                 Controlled rotation around y axis
"CRZ"                 Controlled rotation around z axis
"CX"                  Controlled X gate
"CY"                  Controlled Y gate
"CZ"                  Controlled Z gate
"CS"                  Controlled S gate
"CT"                  Controlled T gate
"CPHASE"              Controlled phase gate
"CNOT"                Controlled NOT gate
"CSIGN"               Same as CPHASE
"QASMU"               U rotation gate used as a primitive in the QASM standard
"BERKELEY"            Berkeley gate
"SWAPalpha"           SWAPalpha gate
"SWAP"                Swap the states of two qubits
"ISWAP"               Swap gate with additional phase for 01 and 10 states
"SQRTSWAP"            Square root of the SWAP gate
"SQRTISWAP"           Square root of the ISWAP gate
"FREDKIN"             Fredkin gate
"TOFFOLI"             Toffoli gate
"GLOBALPHASE"         Global phase
====================  ========================================

For some of the gates listed above, :class:`.QubitCircuit` also has a primitive :func:`.QubitCircuit.resolve_gates()` method that decomposes them into elementary gate sets such as CNOT or SWAP with single-qubit gates (RX, RY and RZ). However, this method is not fully optimized. It is very likely that the depth of the circuit can be further reduced by merging quantum gates. It is required that the gate resolution be carried out before the measurements to the circuit are added.

**Custom Gates**

In addition to these pre-defined gates, QuTiP also allows the user to define their own gate.
The following example shows how to define a customized gate.
The key step is to define a
gate function returning a :class:`qutip.Qobj` and save it in the attribute ``user_gates``.

.. note::

   Available from QuTiP 4.4

.. testcode::

      from qutip_qip.operations import Gate, rx

      def user_gate1(arg_value):
           # controlled rotation X
           mat = np.zeros((4, 4), dtype=np.complex)
           mat[0, 0] = mat[1, 1] = 1.
           mat[2:4, 2:4] = rx(arg_value)
           return Qobj(mat, dims=[[2, 2], [2, 2]])


      def user_gate2():
           # S gate
           mat = np.array([[1.,   0],
                           [0., 1.j]])
           return Qobj(mat, dims=[[2], [2]])

      qc = QubitCircuit(2)
      qc.user_gates = {"CTRLRX": user_gate1,
                       "S"     : user_gate2}

      # qubit 0 controls qubit 1
      qc.add_gate("CTRLRX", targets=[0,1], arg_value=np.pi/2)

      # qubit 1 controls qubit 0
      qc.add_gate("CTRLRX", targets=[1,0], arg_value=np.pi/2)

      # we also add a gate using a predefined Gate object
      g_T = Gate("S", targets=[1])
      qc.add_gate(g_T)
      props = qc.propagators()

      print(props[0])

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
  Qobj data =
  [[1.        +0.j         0.        +0.j         0.        +0.j
    0.        +0.j        ]
   [0.        +0.j         1.        +0.j         0.        +0.j
    0.        +0.j        ]
   [0.        +0.j         0.        +0.j         0.70710678+0.j
    0.        -0.70710678j]
   [0.        +0.j         0.        +0.j         0.        -0.70710678j
    0.70710678+0.j        ]]

.. testcode::

      print(props[1])

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE


  Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
  Qobj data =
  [[1.        +0.j         0.        +0.j         0.        +0.j
    0.        +0.j        ]
   [0.        +0.j         0.70710678+0.j         0.        +0.j
    0.        -0.70710678j]
   [0.        +0.j         0.        +0.j         1.        +0.j
    0.        +0.j        ]
   [0.        +0.j         0.        -0.70710678j 0.        +0.j
    0.70710678+0.j        ]]


.. testcode::

      print(props[2])

**Output**:

.. testoutput::
  :options: +NORMALIZE_WHITESPACE

  Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = False
  Qobj data =
  [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
   [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
   [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
   [0.+0.j 0.+0.j 0.+0.j 0.+1.j]]

.. _quantum_circuit_plots:

Plotting a Quantum Circuit
===================================

A quantum circuit (described above) can directly be plotted using the QCircuit library (https://github.com/CQuIC/qcircuit).
QCiruit is a quantum circuit drawing application and is implemented directly into QuTiP.

More information related to installing these packages is also available in the
installation guide (:ref:`circuit_plot_packages`).

An example code for plotting the example quantum circuit from above is given:

.. code-block:: python

    from qutip_qip.circuit import QubitCircuit
    from qutip_qip.operations import Gate
    # create the quantum circuit
    qc = QubitCircuit(2, num_cbits=1)
    qc.add_gate("CNOT", controls=0, targets=1)
    qc.add_gate("SNOT", targets=1)
    qc.add_gate("ISWAP", targets=[0,1])
    qc.add_measurement("M0", targets=1, classical_store=0)
    # plot the quantum circuit
    qc.png

.. image:: /figures//quantum_circuit_example.png

..
   _This: is a comment, do not test the png generation as it requires additional installation!


Circuit simulation
==================

There are two different ways to simulate the action of quantum circuits using QuTiP:

- The first method utilizes unitary application through matrix products on the input states.
  This method simulates circuits exactly in a deterministic manner. This is achieved through
  :class:`.CircuitSimulator`. A short guide to exact simulation can be
  found at :ref:`qip_simulator`. The teleportation notebook is also useful as an example.

- A different method of circuit simulation employs driving Hamiltonians with the ability to
  simulate circuits in the presence of noise. This can be achieved through the various classes
  in :class:`.device`.A short guide to processors for QIP simulation can be found at :ref:`qip_processor`.
