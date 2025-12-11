_map_gates = {
    "p": "PHASEGATE",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "SNOT",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "swap": "SWAP",
    "u": "QASMU",
}

_map_controlled_gates = {
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "cp": "CPHASE",
}

_ignore_gates = ["id", "barrier"]