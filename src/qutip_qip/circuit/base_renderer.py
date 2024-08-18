"""
Base class Implementation for Renderers
"""

from typing import List

__all__ = ["BaseRenderer"]


class BaseRenderer:
    """
    Base class for rendering quantum circuits with MatRender and TextRender.
    """

    def __init__(self, style):
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
