"""
Base class Implementation for Renderers
"""

from typing import List

__all__ = ["BaseRenderer"]


class BaseRenderer:
    """
    Base class for rendering quantum circuits with MatRender and TextRender.
    """

    def __init__(self):
        """
        Initialize the base renderer with default values.
        Both Renderers should override these attributes as needed.
        """
        self.align_layer = False
        self.qwires = 0
        self.layer_list = []
        self.gate_margin = 0

    # @property
    # @abstractmethod
    # def layer_list(self) -> List[List[float]]:
    #     """
    #     Abstract property for layer list, to be implemented by child classes.
    #     """
    #     pass

    # @property
    # @abstractmethod
    # def gate_margin(self) -> float:
    #     """
    #     Abstract property for gate margin, to be implemented by child classes.
    #     """
    #     pass

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

        xskip = []
        if self.align_layer:
            wire_list = list(range(self.qwires))

        for wire in wire_list:
            xskip.append(sum(self.layer_list[wire][:layer]))

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

        wire_list : List[int]
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.

        xskip : float, optional
            The horizontal value for getting to requested layer. The default is 0.
        """

        for wire in wire_list:
            if len(self.layer_list[wire]) > layer:
                if (
                    self.layer_list[wire][layer]
                    < gate_width + self.gate_margin * 2
                ):
                    self.layer_list[wire][layer] = (
                        gate_width + self.gate_margin * 2
                    )
            else:
                temp = xskip - sum(self.layer_list[wire]) if xskip != 0 else 0
                self.layer_list[wire].append(
                    temp + gate_width + self.gate_margin * 2
                )
