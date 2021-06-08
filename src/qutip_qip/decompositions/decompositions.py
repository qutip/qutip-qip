# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
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
import numbers
from collections.abc import Iterable
from itertools import product, chain
from functools import partial, reduce
from operator import mul

import warnings
import inspect
from copy import deepcopy

import numpy as np
import scipy.sparse as sp

from qutip import (Qobj, identity, qeye, sigmax, sigmay, sigmaz)

# Converts a Qobj or Gate Object to an array
def convert_obj_to_array(input_object):
    if input_object is None:
        raise ValueError("A Qobj or Gate instance must be provided.")
    if isinstance(input_object,Qobj) is True:
        input_object = input_object.full()
    elif isinstance(input_object,Gate) is True:
        input_object = input_object # this function needs to be defined
    else:
        raise TypeError("Accepted input type is either a Qobj or a Gate instance.")

# Convert an array to Qobj or Gate instance
def convert_array_to_obj(input_array, object_type):
    if input_array is None:
        raise ValueError("An array must be provided for conversion.")
    if object_type is None:
        raise ValueError("A Qobj or Gate instance must be provided.")
    if object_type == 'Qobj':
        input_array = Qobj(input_array)
    elif object_type == 'Gate':
        input_array = Gate(input_array) # this function needs to be defined
    else:
        raise TypeError("Specified object type is not a Qobj or Gate instance.")




class Decomposition:
    """Representation of some decomposition function.
    """

    def __init__(self, scheme_name, input_array):
        """
        Create a decomposition with specified parameters.
        """

        self.scheme_name = scheme_name
        self.input_array = input_array

    # list of decomposition schemes - names to be changed as methods are added
    decomposition_schemes = ['scheme1','scheme2']

    # Check is a valid input decomposition scheme is provided
    if scheme_name is None:
        raise NameError("Decomposition scheme name must be provided.")
    elif scheme_name not in decomposition_schemes:
        raise AttributeError("Not a valid decomposition scheme. You could define your own decomposition function via user_defined_decomposition. ")

    # Check if input gate's matrix is of a valid type. If not then converter
    # function tries to convert Qobj / Gate Object to an array
    if isinstance(input_array, np.array) is False:
        try:
            convert_obj_to_array(input_array)
        else:
            raise AssertionError("Input gate should be input as an array, Qobj"+
                            " or Gate instance.")
    input_array = convert_obj_to_array(input_array)

    return scheme_name(input_array)
