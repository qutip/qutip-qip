# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in sourc e and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
from qutip.qobj import Qobj
from qutip_qip.decompositions.general_decompositions import check_input, find_qubits_in_circuit, convert_qobj_gate_to_array
import numpy as np
import pytest
from qutip_qip.operations.gates import z_gate, rx, t_gate

@pytest.mark.parametrize("invalid_input",[np.array([[1,1],[1,1]]),([[1,1],[1,1]]),1.5,3,(1,2,3,4),np.array([[],[]]),([[],[]]),()])
def test_check_input_non_qobj(invalid_input):
    """Checks if correct value is returned or not when the input is not a Qobj.
    """
    with pytest.raises(TypeError,match="The input matrix is not a Qobj."):
        check_input(invalid_input)

@pytest.mark.parametrize("matrix",[Qobj([[1, 2], [3, 4],[5,6]])])
def test_non_square_matrix(matrix):
    """Rectangular Qobj are identified correctly."""
    with pytest.raises(ValueError, match="Input is not a square matrix."):
        check_input(matrix)

@pytest.mark.parametrize("matrix",[Qobj([[1,4,3]]),Qobj([[1]]),Qobj([[2]])])
def test_1d_matrix(matrix):
    """A 1D object is identified as invalid quantum gate"""
    with pytest.raises(ValueError, match="A 1-D Qobj is not a valid quantum gate."):
        check_input(matrix)


@pytest.mark.parametrize("matrix",[Qobj([[1]]),Qobj([[2]]),Qobj([[]])])
def test_1d_matrix(matrix):
    """A 1D object is identified as invalid quantum gate"""
    with pytest.raises(ValueError, match="A 1-D Qobj is not a valid quantum gate."):
        check_input(matrix)

# TO DO : CHECK FOR LARGER NUMBER OF QUBITS
@pytest.mark.parametrize("unitary",[Qobj([[1,0],[0,-1]])])
def test_check_input_non_qobj(unitary):
    """Checks if unitary innput is correctly identified.
    """
    assert(check_input(unitary)==True)

@pytest.mark.parametrize("non_unitary",[Qobj([[1,1],[0,1]])])
def test_check_input_non_qobj(non_unitary):
    """Checks if non-unitary input is correctly identified.
    """
    assert(check_input(non_unitary)==False)


@pytest.mark.parametrize("valid_input",[Qobj([[1,0],[0,-1]]),rx(np.pi/2),z_gate(1),t_gate(1)])
def test_one_qubit_gates(valid_input):
    """Checks if number of qubits the gate will act on are identified properly
    for one qubit gates.
    """
    assert(find_qubits_in_circuit(valid_input)==1)

@pytest.mark.parametrize("valid_input",[rx(np.pi/2,2),z_gate(2),t_gate(2)])
def test_two_qubit_gates(valid_input):
    """Checks if number of qubits the gate will act on are identified properly
    for two qubit gates.
    """
    assert(find_qubits_in_circuit(valid_input)==2)

@pytest.mark.parametrize("valid_input",[rx(np.pi/2,3),z_gate(3),t_gate(3)])
def test_three_qubit_gates(valid_input):
    """Checks if number of qubits the gate will act on are identified properly
    for three qubit gates.
    """
    assert(find_qubits_in_circuit(valid_input)==3)

@pytest.mark.parametrize("valid_input",[Qobj([[1,0,0],[0,1,0],[0,0,1]])])
def test_one_qutrit_gates(valid_input):
    """Checks if qutrit gate is identified.
    """
    with pytest.raises(ValueError,match="Input is operationg on odd dimensional qudit state."):
        find_qubits_in_circuit(valid_input)

@pytest.mark.parametrize("valid_input",[Qobj([[1,0,0],[0,1,0],[0,0,1]]),rx(np.pi/2,3),z_gate(3),t_gate(3)])
def test_one_qutrit_gates(valid_input):
    """Checks if Qobj is converted to a numpy array.
    """
    assert(isinstance(convert_qobj_gate_to_array(valid_input),np.ndarray))
