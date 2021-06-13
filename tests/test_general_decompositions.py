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
import numpy as np
import pytest
from qutip_qip.decompositions.general_decompositions import (
normalize_matrix, check_unitary, extract_global_phase)
from qutip import Qobj

def test_normalize_matrix_valid_input_not_unitary():
    """ Test output when a valid input is provided which is not a unitary.
    """
    normalized_M = normalize_matrix(np.array([[1, 2], [3, 4]]))
    calculated_M = np.array([[0.70711, 1.41421],[2.12132, 2.82843]])
    assert np.array_equal(calculated_M, normalized_M)
    assert check_unitary(calculated_M) == False

@pytest.mark.parametrize("invalid_input", (Qobj([[1],[2],[3],[4],[5]]),(1,2),1.0,10))
@pytest.mark.parametrize("method",[normalize_matrix,extract_global_phase,check_unitary])
def test_method_invalid_input(invalid_input,method):
    """ Test error when Qobj array is provided as input.
    """
    with pytest.raises(TypeError, match="Not a valid input : A Numpy input array must be provided."):
        method(invalid_input)

@pytest.mark.parametrize("method",(normalize_matrix,extract_global_phase,check_unitary))
def test_method_empty_array(method):
    """When an empty array is provided as input."""
    with pytest.raises(ValueError, match="An empty array was provided as input."):
        method(np.array([]))

@pytest.mark.parametrize("input",[np.array([[1,4,3]]),np.array([[1, 2], [3, 4],[5,6]])])
@pytest.mark.parametrize("method",(normalize_matrix,extract_global_phase,check_unitary))
def test_method_non_square_matrix(input,method):
    """Test error is raised when row number and column number of Matrix
    are mismatched."""
    with pytest.raises(ValueError, match="Input must be a square matrix to be a valid gate."):
                    method(input)

@pytest.mark.parametrize("method",(normalize_matrix,extract_global_phase,check_unitary))
def test_method_one_element_array(method):
    """Test proper error is raised when 1x1 matrix is provided as input."""
    with pytest.raises(ValueError, match="Provide a larger array as input."):
        method(np.array([[1]]))

def test_normalize_matrix_zero_determinant():
    """Check if function tries to divide by 0 norm factor.
    """
    with pytest.raises(ZeroDivisionError, match="Determinant of matrix =0."):
        normalize_matrix(np.array([[0, 0], [0, 0]]))

def test_normalize_matrix_normalized_array():
    """Check if function tries to divide by 0 norm factor.
    """
    pauliz = np.array([[1,0],[0,-1]])
    calculated_array = normalize_matrix(pauliz)
    assert all([out==inp] for out, inp in zip(calculated_array,pauliz))


def test_check_unitary():
    """Tests if input is correctly idenitified as unitary. """
    input_array = np.array([[1+1j,1-1j],[1-1j,1+1j]])
    assert check_unitary(input_array) == True

def test_extract_global_phase_non_unitary_input():
    """Tests if a proper error is raised when input to the function is not
    a unitary."""
    with pytest.raises(ValueError, match = "Input array is not a unitary matrix."):
        extract_global_phase(np.array([[1, 2], [3, 4]]))

def test_extract_global_phase_valid_input_phase_comparison():
    """Tests if phase is correctly calculated when input is valid. """

    # Determinant has real part only
    matrix1 = np.multiply(np.array([[1,1],[1,-1]]),1/np.sqrt(2))
    determined_list_of_gates = extract_global_phase(matrix1)
    determined_phase_value = determined_list_of_gates[0]
    actual_phase_value = 1+0j
    assert(determined_phase_value==actual_phase_value)


def test_extract_global_phase_valid_input_output_comparison():
    """Tests if product of outputs is equal to input when input is valid. """

    # Determinant has real part only
    matrix1 = np.multiply(np.array([[1,1],[1,-1]]),1/np.sqrt(2))
    determined_list_of_gates = extract_global_phase(matrix1)
    determined_phase_value = determined_list_of_gates[0]
    calculated_array = np.multiply(determined_list_of_gates[1],determined_phase_value)
    assert all([out==inp] for out, inp in zip(calculated_array,matrix1))


def test_extract_global_phase_valid_input_shape_comparison():
    """Tests if shape of array is unchanged by function. """

    # Determinant has real part only
    matrix1 = np.multiply(np.array([[1,1],[1,-1]]),1/np.sqrt(2))
    determined_list_of_gates = extract_global_phase(matrix1)
    determined_array_shape = np.shape(determined_list_of_gates[1])
    input_array_shape = np.shape(matrix1)
    assert all([out==inp] for out, inp in zip(input_array_shape,determined_array_shape))
