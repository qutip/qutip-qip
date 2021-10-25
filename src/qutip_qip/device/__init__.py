"""
Simulation of quantum hardware.
"""
from .processor import Processor, Model
from .modelprocessor import ModelProcessor
from .spinchain import (
    LinearSpinChain,
    CircularSpinChain,
    SpinChainModel,
)
from .cavityqed import DispersiveCavityQED, CavityQEDModel
from .optpulseprocessor import OptPulseProcessor
from .circuitqed import SCQubits, SCQubitsModel
