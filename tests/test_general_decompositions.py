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
from qutip_qip.decompositions.general_decompositions import normalize_matrix, check_unitary
from qutip import Qobj

def test_normalize_matrix_valid_input_not_unitary():
    """ Test output when a valid input is provided which is not a unitary.
    """
    normalized_M = normalize_matrix(np.array([[1, 2], [3, 4]]))
    calculated_M = np.array([[0.70711, 1.41421],[2.12132, 2.82843]])
    assert np.array_equal(calculated_M, normalized_M)
    assert check_unitary(calculated_M) == False

@pytest.mark.parametrize("invalid_input", (Qobj([[1],[2],[3],[4],[5]]),(1,2)))
def test_normalize_matrix_invalid_input(invalid_input):
    """ Test error when Qobj array is provided as input.
    """
    with pytest.raises(TypeError, match="Not a valid input : A Numpy input array must be provided."):
        normalize_matrix(invalid_input)

def test_normalize_matrix_empty_array():
    """When an empty array is provided as input."""
    with pytest.raises(ValueError, match="An empty array was provided as input."):
        normalize_matrix(np.array([]))


def test_normalize_matrix_non_square_matrix():
    """Test error is raised when row number and column number of Matrix
    are mismatched."""
    with pytest.raises(ValueError, match="Input must be a square matrix to be a valid gate."):
                    normalize_matrix(np.array([[1,4,3]]))
                    normalize_matrix(np.array([[1]]))
                    normalize_matrix(np.array([[1, 2], [3, 4],[5,6]]))

def test_check_unitary():
    """Tests if input is correctly idenitified as unitary. """
    input_array = np.array([[1+1j,1-1j],[1-1j,1+1j]])
    assert check_unitary(input_array) == True
