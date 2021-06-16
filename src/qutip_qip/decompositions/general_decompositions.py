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

from qutip.qobj import Qobj

class ConversionError():
    pass


def extract_global_phase(input_gate):
    """Extracts global phase angle from a valid input quantum gate.

    Parameters
    ----------
    input_gate : :class:`qutip.Qobj`
        The matrix that's supposed to be decomposed should be a Qobj.

    Returns
    -------
    list[phase_angle,new_array]

    phase_angle : float
        Global phase angle :math:`{\\alpha}` in :math:`e^{i {\\alpha}}`.

    new_array : np.array
        Returns an array which is the input gate array multiplied by the
        global phase factor :math:`e^{i {\\alpha}}`.
    """

    # check if input is a qobj
    try:
        isinstance(input_gate,Qobj)
    except:
        raise TypeError("The input matrix is not a Qobj.")

    # check if input is square and a unitary
    input_shape = input_gate.shape
    if input_shape[0] != input_shape[1]:
        raise ValueError("Input is not a square matrix.")
    else:
        unitary_check = Qobj.check_isunitary(input_gate)
        if unitary_check == False:
            raise ValueError("Input is not a unitary quantum gate.")

    # convert input qobj to a numpy array
    input_to_array = Qobj.full(input_gate)
    try:
        isinstance(input_to_array, np.ndarray)
    except:
        raise ConversionError("Input Qobj could not be converted to a numpy array.")

    input_determinant = np.linalg.det(input_to_array)
    # since there's a check for input being a unitary or not, there's no error
    # being raised 0 determinant.
    real_part = np.real(input_determinant)
    im_part = -np.imag(input_determinant)


    phase_angle = (1/input_shape[0])*np.arctan2(im_part,real_part)

    return(phase_angle)
