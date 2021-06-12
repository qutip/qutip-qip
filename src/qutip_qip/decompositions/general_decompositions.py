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

def normalize_matrix(input_array)-> np.array:
    """ Checks if the input gate's array is normalized or not. If not, makes
    sure the input has been normalized.

    This function will also check if the input is a valid array and a valid
    quantum gate.

     Args:
        input_array : Matrix of a gate in Numpy array form.
    """
    if not isinstance(input_array, np.ndarray):
        raise TypeError("Not a valid input : A Numpy input array must be provided.")


    if isinstance(input_array, np.ndarray):
        if input_array.size==0:
            raise ValueError("An empty array was provided as input.")

        input_matrix_rows = input_array.shape[0]
        input_matrix_columns = input_array.shape[1]
        if input_matrix_rows !=input_matrix_columns:
            raise ValueError("Input must be a square matrix to be a valid gate.")

        if np.linalg.det(input_array) != 1:
            if np.linalg.det(input_array) ==0:
                raise ZeroDivisionError("Determinant of matrix =0.")
            norm_factor = float(1/np.abs(np.linalg.det(input_array)))**0.5
            input_array = np.around(norm_factor*input_array,5)
        else:
            input_array = input_array

        return(input_array)

# Note this function is defined for qobj, re-defined here for a numpy array.
# Accessing individual elements of Qobj array could be problematic.
def check_unitary(input_array)-> bool:
    """Checks if the input matrix is unitary or not.

    Args:
       input_array : Matrix of a gate in Numpy array form.
    """
    input_array = normalize_matrix(input_array)
    identity_matrix = np.eye(input_array.shape[0])
    input_array_dagger = input_array.conj().T
    check_unitary_left = np.allclose(identity_matrix, np.dot(input_array_dagger,input_array))
    check_unitary_right = np.allclose(identity_matrix, np.dot(input_array,input_array_dagger))
    if check_unitary_left != check_unitary_right:
        raise ArithmeticError("Unitary product assertions do not match.")
    check_unitary = check_unitary_left
    return(check_unitary)

def extract_global_phase(input_array):
    """Express input array as a product of some global phase factor and a special
    unitary matrix array (returned in the form of a list containing `phasegate`
    and some other special unitary array).

    Args:
       input_array : Matrix of a gate in Numpy array form.
    """
    if check_unitary == True:
        input_array = normalize_matrix(input_array)
        
