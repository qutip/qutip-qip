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
import cmath

from qutip.qobj import Qobj

class ConversionError():
    pass


def check_input(input_gate):
    """Verifies input is a valid quantum gate.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    bool
        If the input is a valid quantum gate matrix, the function returns "True".
        In the case of "False" being returned, this function will ensure no
        decomposition scheme can proceed.
    """
    # check if input is a qobj
    qobj_check = isinstance(input_gate,Qobj)
    if qobj_check == False:
        raise TypeError("The input matrix is not a Qobj.")
    else:
        # check if input is square and a unitary
        input_shape = input_gate.shape
        if input_shape[0]==1:
            raise ValueError("A 1-D Qobj is not a valid quantum gate.")
        if input_shape[0] != input_shape[1]:
            raise ValueError("Input is not a square matrix.")
        else:
            unitary_check = Qobj.check_isunitary(input_gate)
            return unitary_check



def find_qubits_in_circuit(input_gate):
    """Based on the shape of the unitary input gate, determines the number of
    qubits the input gate will act on.

    TO DO: CHANGE THE LOOP FOR QUDITS (CURRENTLY ONLY WORKS FOR QUBITS)

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    number_of_qubits: int
        Returns an integer value for the number of qubits
    """
    input_check_bool = check_input(input_gate)
    if input_check_bool == True:
        input_shape = input_gate.shape
        if input_shape[0]%2==0.0: # won't be 2 for higher qudits
        # TO DO : if d=4 (qu4it) then it will still appear as a d=2 i.e. qubit gate
        # change to powers of 2
            number_of_qubits = np.log(input_shape[0])/np.log(2)
        else:
            raise ValueError("Input is operationg on odd dimensional qudit state.")
        return(int(number_of_qubits))
    else:
        raise ValueError("Input is not unitary.")

def convert_qobj_gate_to_array(input_gate):
    """Converts a valid unitary quantum gate to a numpy array.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    input_gate : `np.array`
        The input is returned as a converted numpy array.
    """
    input_check_bool = check_input(input_gate)
    if input_check_bool == True:
        input_to_array = Qobj.full(input_gate)
        try:
            isinstance(input_to_array, np.ndarray)
        except:
            raise ConversionError("Input Qobj could not be converted to a numpy array.")
        return(input_to_array)
    else:
        raise ValueError("Input is not unitary.")
