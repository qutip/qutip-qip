import numpy as np

from qutip import QobjEvo, Qobj, identity
from qutip_qip.operations import expand_operator


class EvoElement:
    """
    The class object saving the information of one evolution element.
    Each dynamic element is characterized by four variables:
    ``qobj``, ``targets``, ``tlist`` and ``coeff``.

    For documentation and use instruction of the attributes, please
    refer to :class:`.Pulse`.
    """

    def __init__(
        self,
        qobj: Qobj,
        targets: list[int],
        tlist: list[list[float]] | None = None,
        coeff: list[float] | bool | None = None,
    ) -> None:
        self.qobj = qobj
        self.targets = targets
        self.tlist = tlist
        self.coeff = coeff

    def get_qobj(self, dims: int | list[int]) -> Qobj:
        """
        Get the `Qobj` representation of the element. If `qobj` is None,
        a zero :class:`qutip.Qobj` with the corresponding dimension is
        returned.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        qobj : :class:`qutip.Qobj`
            The operator of this element.
        """
        if isinstance(dims, (int, np.integer)):
            dims = [2] * dims

        if self.qobj is None:
            qobj = identity(dims[0]) * 0.0
            targets = 0
        else:
            qobj = self.qobj
            targets = self.targets

        return expand_operator(qobj, dims=dims, targets=targets)

    def _get_qobjevo_helper(
        self,
        spline_kind: str,
        dims: int | list[int],
    ) -> QobjEvo:
        """
        Please refer to `_Evoelement.get_qobjevo` for documentation.
        """
        mat = self.get_qobj(dims)
        if self.tlist is None and self.coeff is None:
            qu = QobjEvo(mat) * 0.0

        elif isinstance(self.coeff, bool):
            if self.coeff:
                if self.tlist is None:
                    qu = QobjEvo(mat, tlist=self.tlist)
                else:
                    qu = QobjEvo(
                        [mat, np.ones(len(self.tlist))], tlist=self.tlist
                    )
            else:
                qu = QobjEvo(mat * 0.0, tlist=self.tlist)

        else:
            if spline_kind == "cubic":
                qu = QobjEvo(
                    [mat, self.coeff],
                    tlist=self.tlist,
                )
            elif spline_kind == "step_func":
                if len(self.coeff) == len(self.tlist) - 1:
                    self.coeff = np.concatenate([self.coeff, [0.0]])
                qu = QobjEvo([mat, self.coeff], tlist=self.tlist, order=0)
            else:
                # The spline will follow other pulses or
                # use the default value of QobjEvo
                raise ValueError("The pulse has an unknown spline type.")
        return qu

    def get_qobjevo(
        self,
        spline_kind: str,
        dims: int | list[int],
    ) -> QobjEvo:
        """
        Get the `QobjEvo` representation of the evolution element.
        If both `tlist` and ``coeff`` are None, treated as zero matrix.
        If ``coeff=True`` and ``tlist=None``,
        treated as time-independent operator.

        Parameters
        ----------
        spline_kind: str
            Type of the coefficient interpolation.
            "step_func" or "cubic"

            -"step_func":
            The coefficient will be treated as a step function.
            E.g. ``tlist=[0,1,2]`` and ``coeff=[3,2]``, means that the
            coefficient is 3 in t=[0,1) and 2 in t=[2,3). It requires
            ``len(coeff)=len(tlist)-1`` or ``len(coeff)=len(tlist)``, but
            in the second case the last element of ``tlist`` has no effect.

            -"cubic": Use cubic interpolation for the coefficient. It requires
            ``len(coeff)=len(tlist)``
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        qobjevo: :class:`qutip.QobjEvo`
            The `QobjEvo` representation of the evolution element.
        """
        try:
            return self._get_qobjevo_helper(spline_kind, dims=dims)
        except Exception as err:
            print(
                "The Evolution element went wrong was\n {}".format(str(self))
            )
            raise (err)

    def __str__(self) -> str:
        return str(
            {
                "qobj": self.qobj,
                "targets": self.targets,
                "tlist": self.tlist,
                "coeff": self.coeff,
            }
        )


def merge_qobjevo(
    qobjevo_list: list[tuple[Qobj, QobjEvo]], full_tlist=None
) -> tuple[Qobj, QobjEvo]:
    """
    Combine a list of `:class:qutip.QobjEvo` into one,
    different tlist will be merged.
    """
    # no qobjevo
    if not qobjevo_list:
        raise ValueError("qobjevo_list is empty.")

    # FIXME full_tlist is unused
    return sum([op for op in qobjevo_list if isinstance(op, (Qobj, QobjEvo))])
