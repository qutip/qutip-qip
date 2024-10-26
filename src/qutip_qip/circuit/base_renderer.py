"""
Base class Implementation for Renderers
"""

from dataclasses import dataclass
from typing import Union, Optional, List, Dict

from .color_theme import qutip, light, dark, modern


__all__ = ["BaseRenderer", "StyleConfig"]


@dataclass
class StyleConfig:
    """
    Dataclass to store the style configuration for circuit customization.

    Parameters
    ----------
    dpi : int, optional
        DPI of the figure. The default is 150.

    fontsize : int, optional
        Fontsize control at circuit level, including tile and wire labels. The default is 10.

    end_wire_ext : int, optional
        Extension of the wire at the end of the circuit. The default is 2.
        Available to TextRender and MatRender.

    padding : float, optional
        Padding between the circuit and the figure border. The default is 0.3.

    gate_margin : float, optional
        Margin space left on each side of the gate. The default is 0.15.

    wire_sep : float, optional
        Separation between the wires. The default is 0.5.

    layer_sep : float, optional
        Separation between the layers. The default is 0.5.

    gate_pad : float, optional
        Padding between the gate and the gate label. The default is 0.05.
        Available to TextRender and MatRender.

    label_pad : float, optional
        Padding between the wire label and the wire. The default is 0.1.

    bulge : Union[str, bool], optional
        Bulge style of the gate. Renders non-bulge gates if False. The default is True.

    align_layer : bool, optional
        Align the layers of the gates across different wires. The default is False.
        Available to TextRender and MatRender.

    theme : Optional[Union[str, Dict]], optional
        Color theme of the circuit. The default is "qutip".
        The available themes are 'qutip', 'light', 'dark' and 'modern'.

    title : Optional[str], optional
        Title of the circuit. The default is None.

    bgcolor : Optional[str], optional
        Background color of the circuit. The default is None.

    color : Optional[str], optional
        Controls color of acsent elements (eg. cross sign in the target node)
        and set as deafult color of gate-label. Can be overwritten by gate specific color.
        The default is None.

    wire_label : Optional[List], optional
        Labels of the wires. The default is None.
        Available to TextRender and MatRender.

    wire_color : Optional[str], optional
        Color of the wires. The default is None.
    """

    dpi: int = 150
    fontsize: int = 10
    end_wire_ext: int = 2
    padding: float = 0.3
    gate_margin: float = 0.15
    wire_sep: float = 0.5
    layer_sep: float = 0.5
    gate_pad: float = 0.05
    label_pad: float = 0.1
    bulge: Union[str, bool] = True
    align_layer: bool = False
    theme: Optional[Union[str, Dict]] = "qutip"
    title: Optional[str] = None
    bgcolor: Optional[str] = None
    color: Optional[str] = None
    wire_label: Optional[List] = None
    wire_color: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.bulge, bool):
            self.bulge = "round4" if self.bulge else "square"

        self.measure_color = "#000000"
        if self.theme == "qutip":
            self.theme = qutip
        elif self.theme == "light":
            self.theme = light
        elif self.theme == "dark":
            self.theme = dark
            self.measure_color = "#FFFFFF"
        elif self.theme == "modern":
            self.theme = modern
        else:
            raise ValueError(
                f"""Invalid theme: {self.theme},
                Must be selectec from 'qutip', 'light', 'dark', or 'modern'.
                """
            )

        self.bgcolor = self.bgcolor or self.theme["bgcolor"]
        self.color = self.color or self.theme["color"]
        self.wire_color = self.wire_color or self.theme["wire_color"]


class BaseRenderer:
    """
    Base class for rendering quantum circuits with MatRender and TextRender.
    """

    def __init__(self, style: StyleConfig):
        """
        Initialize the base renderer with default values.
        Both Renderers should override these attributes as needed.
        """

        self._qwires = 0
        self._layer_list = []
        self.style = style

    def _get_xskip(self, wire_list: List[int], layer: int) -> float:
        """
        Get the xskip (horizontal value for getting to requested layer) for the gate to be plotted.

        Parameters
        ----------
        wire_list : List[int]
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.

        Returns
        -------
        float
            The maximum xskip value needed to reach the specified layer.
        """

        if self.style.align_layer:
            wire_list = list(range(self._qwires))

        xskip = []
        for wire in wire_list:
            xskip.append(sum(self._layer_list[wire][:layer]))

        return max(xskip)

    def _manage_layers(
        self,
        gate_width: float,
        wire_list: List[int],
        layer: int,
        xskip: float = 0,
    ) -> None:
        """
        Manages and updates the layer widths according to the gate's width just plotted.

        Parameters
        ----------
        gate_width : float
            The width of the gate to be plotted.

        wire_list : list
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.

        xskip : float, optional
            The horizontal value for getting to requested layer. The default is 0.
        """

        for wire in wire_list:
            # check if requested layer exists for the wire
            if len(self._layer_list[wire]) > layer:
                # check if the layer width is greater than new layer width
                if (
                    self._layer_list[wire][layer]
                    < gate_width + self.style.gate_margin * 2
                ):
                    # update with new layer width
                    self._layer_list[wire][layer] = (
                        gate_width + self.style.gate_margin * 2
                    )
            else:
                # add layer width: new layer width + missing layer widths if exits
                temp = xskip - sum(self._layer_list[wire]) if xskip != 0 else 0
                self._layer_list[wire].append(
                    temp + gate_width + self.style.gate_margin * 2
                )
