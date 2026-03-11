"""
Color theme for different gates in the circuit diagram.
"""

qutip: dict[str, str] = {
    "bgcolor": "#FFFFFF",  # White
    "color": "#FFFFFF",  # White
    "wire_color": "#000000",  # Black
    "default_gate": "#000000",  # Black
    "IDLE": "#FFFFFF",  # White
    "X": "#CB4BF9",  # Medium Orchid
    "Y": "#CB4BF9",  # Medium Orchid
    "Z": "#CB4BF9",  # Medium Orchid
    "H": "#6270CE",  # Medium Slate Blue
    "SNOT": "#6270CE",  # Medium Slate Blue
    "S": "#254065",  # Dark Slate Blue
    "T": "#254065",  # Dark Slate Blue
    "Sdag": "#254065",  # Dark Slate Blue
    "Tdag": "#254065",  # Dark Slate Blue
    "SQRTX": "#2CAC70",  # Green
    "SQRTXdag": "#2CAC70",  # Green
    "SQRTNOT": "#2CAC70",  # Green
    "RX": "#5EBDF8",  # Light Sky Blue
    "RY": "#5EBDF8",  # Light Sky Blue
    "RZ": "#5EBDF8",  # Light Sky Blue
    "PHASE": "#456DB2",  # Steel Blue
    "R": "#456DB2",  # Steel Blue
    "QASMU": "#456DB2",  # Steel Blue
    "SWAP": "#3B3470",  # Indigo
    "SQRTSWAP": "#3B3470",  # Indigo
    "SQRTSWAPdag": "#3B3470",  # Indigo
    "ISWAP": "#3B3470",  # Indigo
    "ISWAPdag": "#3B3470",  # Indigo
    "SQRTISWAP": "#3B3470",  # Indigo
    "SQRTISWAPdag": "#3B3470",  # Indigo
    "BERKELEY": "#3B3470",  # Indigo
    "BERKELEYdag": "#3B3470",  # Indigo
    "SWAPALPHA": "#7648CB",  # Dark Orchid
    "MS": "#7648CB",  # Dark Orchid
    "RZX": "#7648CB",  # Dark Orchid
    "CX": "#9598F5",  # Light Slate Blue
    "CY": "#9598F5",  # Light Slate Blue
    "CZ": "#9598F5",  # Light Slate Blue
    "CH": "#9598F5",  # Light Slate Blue
    "CS": "#9598F5",  # Light Slate Blue
    "CSdag": "#9598F5",  # Light Slate Blue
    "CT": "#9598F5",  # Light Slate Blue
    "CTdag": "#9598F5",  # Light Slate Blue
    "CRX": "#A66DDF",  # Medium Purple
    "CRY": "#A66DDF",  # Medium Purple
    "CRZ": "#A66DDF",  # Medium Purple
    "CPHASE": "#A66DDF",  # Medium Purple
    "CQASMU": "#A66DDF",  # Medium Purple
    "TOFFOLI": "#3B3470",  # Indigo
    "FREDKIN": "#7648CB",  # Dark Orchid
}

light: dict[str, str] = {
    "bgcolor": "#EEEEEE",  # Light Gray
    "color": "#000000",  # Black
    "wire_color": "#000000",  # Black
    "default_gate": "#D8CDAF",  # Bit Dark Beige
    "IDLE": "#FFFFFF",  # White
    "X": "#F4A7B9",  # Light Pink
    "Y": "#F4A7B9",  # Light Pink
    "Z": "#F4A7B9",  # Light Pink
    "H": "#A3C1DA",  # Light Blue
    "SNOT": "#A3C1DA",  # Light Blue
    "S": "#D3E2EE",  # Very Light Blue
    "T": "#D3E2EE",  # Very Light Blue
    "Sdag": "#D3E2EE",  # Very Light Blue
    "Tdag": "#D3E2EE",  # Very Light Blue
    "SQRTX": "#E1E0BA",  # Light Yellow
    "SQRTXdag": "#E1E0BA",  # Light Yellow
    "SQRTNOT": "#E1E0BA",  # Light Yellow
    "RX": "#B3E6E4",  # Light Teal
    "RY": "#B3E6E4",  # Light Teal
    "RZ": "#B3E6E4",  # Light Teal
    "PHASE": "#D5E0F2",  # Light Slate Blue
    "R": "#D5E0F2",  # Light Slate Blue
    "QASMU": "#D5E0F2",  # Light Slate Blue
    "SWAP": "#FFB6B6",  # Lighter Coral Pink
    "SQRTSWAP": "#FFB6B6",  # Lighter Coral Pink
    "SQRTSWAPdag": "#FFB6B6",  # Lighter Coral Pink
    "ISWAP": "#FFB6B6",  # Lighter Coral Pink
    "ISWAPdag": "#FFB6B6",  # Lighter Coral Pink
    "SQRTISWAP": "#FFB6B6",  # Lighter Coral Pink
    "SQRTISWAPdag": "#FFB6B6",  # Lighter Coral Pink
    "BERKELEY": "#FFB6B6",  # Lighter Coral Pink
    "BERKELEYdag": "#FFB6B6",  # Lighter Coral Pink
    "SWAPALPHA": "#CDC1E8",  # Light Purple
    "MS": "#CDC1E8",  # Light Purple
    "RZX": "#CDC1E8",  # Light Purple
    "CX": "#E0E2F7",  # Very Light Indigo
    "CY": "#E0E2F7",  # Very Light Indigo
    "CZ": "#E0E2F7",  # Very Light Indigo
    "CH": "#E0E2F7",  # Very Light Indigo
    "CS": "#E0E2F7",  # Very Light Indigo
    "CSdag": "#E0E2F7",  # Very Light Indigo
    "CT": "#E0E2F7",  # Very Light Indigo
    "CTdag": "#E0E2F7",  # Very Light Indigo
    "CRX": "#D6C9E8",  # Light Muted Purple
    "CRY": "#D6C9E8",  # Light Muted Purple
    "CRZ": "#D6C9E8",  # Light Muted Purple
    "CPHASE": "#D6C9E8",  # Light Slate Blue
    "CQASMU": "#D6C9E8",  # Light Slate Blue
    "TOFFOLI": "#E6CCE6",  # Soft Lavender
    "FREDKIN": "#E6CCE6",  # Soft Lavender
}

dark: dict[str, str] = {
    "bgcolor": "#000000",  # Black
    "color": "#000000",  # Black
    "wire_color": "#989898",  # Dark Gray
    "default_gate": "#D8BFD8",  # (Thistle)
    "IDLE": "#FFFFFF",  # White
    "X": "#9370DB",  # Medium Purple
    "Y": "#9370DB",  # Medium Purple
    "Z": "#9370DB",  # Medium Purple
    "S": "#B0E0E6",  # Powder Blue
    "H": "#AFEEEE",  # Pale Turquoise
    "SNOT": "#AFEEEE",  # Pale Turquoise
    "T": "#B0E0E6",  # Powder Blue
    "Sdag": "#B0E0E6",  # Powder Blue
    "Tdag": "#B0E0E6",  # Powder Blue
    "SQRTX": "#718520",  # Olive Yellow
    "SQRTXdag": "#718520",  # Olive Yellow
    "SQRTNOT": "#718520",  # Olive Yellow
    "RX": "#87CEEB",  # Sky Blue
    "RY": "#87CEEB",  # Sky Blue
    "RZ": "#87CEEB",  # Sky Blue
    "PHASE": "#8A2BE2",  # Blue Violet
    "R": "#8A2BE2",  # Blue Violet
    "QASMU": "#8A2BE2",  # Blue Violet
    "SWAP": "#BA55D3",  # Medium Orchid
    "SQRTSWAP": "#BA55D3",  # Medium Orchid
    "SQRTSWAPdag": "#BA55D3",  # Medium Orchid
    "ISWAP": "#BA55D3",  # Medium Orchid
    "ISWAPdag": "#BA55D3",  # Medium Orchid
    "SQRTISWAP": "#BA55D3",  # Medium Orchid
    "SQRTISWAPdag": "#BA55D3",  # Medium Orchid
    "BERKELEY": "#BA55D3",  # Medium Orchid
    "BERKELEYdag": "#BA55D3",  # Medium Orchid
    "SWAPALPHA": "#6A5ACD",  # Slate Blue
    "MS": "#6A5ACD",  # Slate Blue
    "RZX": "#6A5ACD",  # Slate Blue
    "CX": "#4682B4",  # Steel Blue
    "CY": "#4682B4",  # Steel Blue
    "CZ": "#4682B4",  # Steel Blue
    "CH": "#4682B4",  # Steel Blue
    "CS": "#4682B4",  # Steel Blue
    "CSdag": "#4682B4",  # Steel Blue
    "CT": "#4682B4",  # Steel Blue
    "CTdag": "#4682B4",  # Steel Blue
    "CRX": "#7B68EE",  # Medium Slate Blue
    "CRY": "#7B68EE",  # Medium Slate Blue
    "CRZ": "#7B68EE",  # Medium Slate Blue
    "CPHASE": "#DA70D6",  # Orchid
    "CQASMU": "#DA70D6",  # Orchid
    "TOFFOLI": "#43414F",  # Dark Gray
    "FREDKIN": "#43414F",  # Dark Gray
}


modern: dict[str, str] = {
    "bgcolor": "#FFFFFF",  # White
    "color": "#FFFFFF",  # White
    "wire_color": "#000000",  # Black
    "default_gate": "#ED9455",  # Slate Orange
    "IDLE": "#FFFFFF",  # White
    "X": "#4A5D6D",  # Dark Slate Blue
    "Y": "#4A5D6D",  # Dark Slate Blue
    "Z": "#4A5D6D",  # Dark Slate Blue
    "H": "#C25454",  # Soft Red
    "SNOT": "#C25454",  # Soft Red
    "S": "#2C3E50",  # Very Dark Slate Blue
    "T": "#2C3E50",  # Very Dark Slate Blue
    "Sdag": "#2C3E50",  # Very Dark Slate Blue
    "Tdag": "#2C3E50",  # Very Dark Slate Blue
    "SQRTX": "#D2E587",  # Yellow
    "SQRTXdag": "#D2E587",  # Yellow
    "SQRTNOT": "#D2E587",  # Yellow
    "RX": "#2F4F4F",  # Dark Slate Teal
    "RY": "#2F4F4F",  # Dark Slate Teal
    "RZ": "#2F4F4F",  # Dark Slate Teal
    "PHASE": "#5E7D8B",  # Dark Slate Blue
    "R": "#5E7D8B",  # Dark Slate Blue
    "QASMU": "#5E7D8B",  # Dark Slate Blue
    "SWAP": "#6A9ACD",  # Slate Blue
    "SQRTSWAP": "#6A9ACD",  # Slate Blue
    "SQRTSWAPdag": "#6A9ACD",  # Slate Blue
    "ISWAP": "#6A9ACD",  # Slate Blue
    "ISWAPdag": "#6A9ACD",  # Slate Blue
    "SQRTISWAP": "#6A9ACD",  # Slate Blue
    "SQRTISWAPdag": "#6A9ACD",  # Slate Blue
    "BERKELEY": "#6A9ACD",  # Slate Blue
    "BERKELEYdag": "#6A9ACD",  # Slate Blue
    "SWAPALPHA": "#4A5D6D",  # Dark Slate Blue
    "MS": "#4A5D6D",  # Dark Slate Blue
    "RZX": "#4A5D6D",  # Dark Slate Blue
    "CX": "#5D8AA8",  # Medium Slate Blue
    "CY": "#5D8AA8",  # Medium Slate Blue
    "CZ": "#5D8AA8",  # Medium Slate Blue
    "CH": "#5D8AA8",  # Medium Slate Blue
    "CS": "#5D8AA8",  # Medium Slate Blue
    "CSdag": "#5D8AA8",  # Medium Slate Blue
    "CT": "#5D8AA8",  # Medium Slate Blue
    "CTdag": "#5D8AA8",  # Medium Slate Blue
    "CRX": "#6C5B7B",  # Dark Lavender
    "CRY": "#6C5B7B",  # Dark Lavender
    "CRZ": "#6C5B7B",  # Dark Lavender
    "CPHASE": "#4A4A4A",  # Dark Gray
    "CQASMU": "#4A4A4A",  # Dark Gray
    "TOFFOLI": "#4A5D6D",  # Dark Slate Blue
    "FREDKIN": "#4A5D6D",  # Dark Slate Blue
}
