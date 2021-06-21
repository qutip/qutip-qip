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
from .general_decompositions import (check_input, find_qubits_in_circuit,
                                    convert_qobj_gate_to_array, extract_global_phase)

from ..operations import (ry, rz, globalphase, x_gate)

def ZY_decomposition(input_gate):
    """Decomposes input quantum gate into a product of Ry and Rz.

    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    num_of_qubits = find_qubits_in_circuit(input_gate)
    if num_of_qubits != 1:
        raise AttributeError("Input is not a 1-qubit gate.")
    else:
        if check_input(input_gate) == True:
            input_list_of_phase_array = extract_global_phase(input_gate)
        else:
            raise ValueError("Input is not unitary.")

        phase_angle = input_list_of_phase_array[0]
        num_of_qubits = input_list_of_phase_array[1]
        input_array = input_list_of_phase_array[2]

        # Change labels of variables
        # Currently, it is all consistent with the calculations pdf.
        a = input_array[0][0]
        b = input_array[0][1]

        # Find the angles for Rz and Ry
        alpha = cmath.phase(complex(-np.imag(a),np.real(a))) + cmath.phase(complex(-np.imag(b),np.real(b)))
        beta = cmath.phase(complex(-np.imag(a),np.real(a))) - cmath.phase(complex(-np.imag(b),np.real(b)))
        theta = cmath.phase(np.sqrt(cmath.abs(b)/cmath.abs(a)))

        # Return a list of the gate arrays
        gate_1 = rz(alpha)
        gate_2 = ry(theta)
        gate_3 = rz(beta)
        global_phase_gate = globalphase(phase_angle)

        output_list = [global_phase_gate,gate_1,gate_2,gate_3]

def ABC_decomposition(input_gate):
    """Decomposes input quantum gate into a product of Pauli X, Ry and Rz.

    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.
    """
    num_of_qubits = find_qubits_in_circuit(input_gate)
    if num_of_qubits != 1:
        raise AttributeError("Input is not a 1-qubit gate.")
    else:
        if check_input(input_gate) == True:
            input_list_of_phase_array = extract_global_phase(input_gate)
        else:
            raise ValueError("Input is not unitary.")

        phase_angle = input_list_of_phase_array[0]
        num_of_qubits = input_list_of_phase_array[1]
        input_array = input_list_of_phase_array[2]

        # Change labels of variables
        # Currently, it is all consistent with the calculations pdf.
        a = input_array[0][0]
        b = input_array[0][1]

        # Find the angles for Rz and Ry
        alpha = cmath.phase(complex(-np.imag(a),np.real(a))) + cmath.phase(complex(-np.imag(b),np.real(b)))
        beta = cmath.phase(complex(-np.imag(a),np.real(a))) - cmath.phase(complex(-np.imag(b),np.real(b)))
        theta = cmath.phase(np.sqrt(cmath.abs(b)/cmath.abs(a)))

        # Return a list of the gate arrays
        A_1 = rz(alpha)
        A_2 = ry(theta/2)
        B_1 = ry(-theta/2)
        B_2 = rz(-(alpha+beta)/2)
        C = rz((-alpha+beta)/2)
        global_phase_gate = globalphase(phase_angle)

        output_list = [global_phase_gate, A_1, A_2,x_gate, B_1, B_2, x_gate, C]


def single_qubit_decomposition(input_choice):
    """ Decomposes an arbitrary input 1 - qubit gate based on the user's choice.
    """

    if (input_choice == 'optimize_decomposition') or (input_choice = 'ZY_decomposition'):
        calculated_decomposition = ZY_decomposition(input_choice)
    elif input_choice == 'ABC_decomposition':
        calculated_decomposition = ABC_decomposition(input_choice)
    else:
        raise NameError("Decomposition method not defined.")
    return(calculated_decomposition)
