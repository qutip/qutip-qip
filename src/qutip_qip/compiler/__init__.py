"""Compilers for the hardware models in :obj:`device`"""

import warnings
from .instruction import PulseInstruction
from .scheduler import Scheduler
from .gatecompiler import GateCompiler
from .spinchaincompiler import SpinChainCompiler
from .cavityqedcompiler import CavityQEDCompiler
from .circuitqedcompiler import SCQubitsCompiler


class Instruction(PulseInstruction):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Instruction is deprecated and has been renamed to PulseInstruction. "
            "Use PulseInstruction instead.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "PulseInstruction",
    "Instruction",
    "Scheduler",
    "GateCompiler",
    "SpinChainCompiler",
    "CavityQEDCompiler",
    "SCQubitsCompiler",
]
