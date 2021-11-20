from collections import deque
from copy import deepcopy
from functools import cmp_to_key
from random import shuffle

from ..circuit import QubitCircuit
from ..operations import Gate
from .instruction import Instruction


class InstructionsGraph:
    """
    A directed acyclic graph (DAG) representation
    of the quantum instruction dependency.
    An example is Fig3(b) in https://doi.org/10.1117/12.666419.
    It contains methods of generating the instruction dependency graph,
    a list-schedule algorithm to find the topological order
    and the computation of the distance in the weighted graph
    (circuit latency).

    It uses the `Instruction` object as a representation of node
    and adds the following attributes to it:

    predecessors, successors: dependency arrow of the DAG
    distance_to_start, distance_to_end: longest distance to the start and end

    Parameters
    ----------
    instructions: list
        A list of instructions

    Attributes
    ----------
    nodes: list
        The input list of instruction with additional graph information.
    start, end: list
        List of indices of nodes connected to the start or end nodes.
    """

    def __init__(self, instructions):
        instructions = deepcopy(instructions)
        self.nodes = []
        for instruction in instructions:
            if isinstance(instruction, Gate):
                self.nodes.append(Instruction(instruction))
            else:
                self.nodes.append(instruction)
        for node in self.nodes:
            if node.duration is None:
                node.duration = 1
        self.start = None
        self.end = None

    def generate_dependency_graph(self, commuting):
        """
        Generate the instruction dependency graph.
        It modifies the class attribute `nodes`, where each element (node)
        is an `Instruction`.
        The graph is represented by attributes `predecessors` and
        `successors`, each a set of indices
        pointing to the position of the target node in the nodes list.

        The graph preserves the mobility of the gates,
        i.e. if two gates commute with each other,
        such as ``CNOT 2, 3`` and ``CNOT 2, 1``,
        there should be no dependency arrow between them.
        Because of this, the generated graph does not consider
        the hardware constraints,
        e.g. two commuting gates addressing the same qubits
        cannot be executed at the same time.
        A dependency arrow between Instruction 1 and instruction 2
        means that they do not commute.
        However, the converse does not hold because we do not need
        gate1->gate3 if we already have gate1->gate2->gate3.

        Parameters
        ----------
        commuting: function
            A Python function that determines if two gates commute,
            given that their used qubits overlap.
        """
        # initialize the graph
        for node in self.nodes:
            node.predecessors = set()
            node.successors = set()

        num_qubits = (
            max(
                set().union(
                    *[instruction.used_qubits for instruction in self.nodes]
                )
            )
            + 1
        )
        # Build the dependency graph with the constraint that
        # one qubit can not be acted on by two operations, except if
        # they commute with each other.
        qubits_cycle_last = [set() for i in range(num_qubits)]
        qubits_cycle_current = [set() for i in range(num_qubits)]

        # Generate instruction (gate) dependency for each qubit
        for gate_index, instruction in enumerate(self.nodes):
            for qubit in instruction.used_qubits:
                dependent = False
                for dependent_ind in qubits_cycle_current[qubit]:
                    if not commuting(gate_index, dependent_ind, self.nodes):
                        dependent = True
                # Assume that if A, B and C use the same qubit, we have
                # [A,B]=0, [B,C]=0 -> [A,C]=0.
                # Then we can conclude that
                # all gates in the current cycle depends on the previous cycle.
                if dependent:
                    self._add_dependency(
                        qubits_cycle_last[qubit], qubits_cycle_current[qubit]
                    )
                    qubits_cycle_last[qubit] = qubits_cycle_current[qubit]
                    qubits_cycle_current[qubit] = {gate_index}
                else:
                    qubits_cycle_current[qubit].add(gate_index)

        for qubit in range(num_qubits):
            self._add_dependency(
                qubits_cycle_last[qubit], qubits_cycle_current[qubit]
            )

        # Find start and end nodes of the graph
        start = []
        end = []
        for i, instruction in enumerate(self.nodes):
            if not instruction.successors:
                end.append(i)
            if not instruction.predecessors:
                start.append(i)
        self.start = start
        self.end = end

    def _add_dependency(self, gates_set_last, gates_set_current):
        for instruction_ind1 in gates_set_last:
            for instruction_ind2 in gates_set_current:
                self.nodes[instruction_ind1].successors.add(instruction_ind2)
                self.nodes[instruction_ind2].predecessors.add(instruction_ind1)

    def reverse_graph(self):
        """
        Reverse the graph.
        The start node becomes the end node
        Predecessors and successors of each node are exchanged.
        """
        for node in self.nodes:
            node.predecessors, node.successors = (
                node.successors,
                node.predecessors,
            )
            try:
                node.distance_to_start, node.distance_to_end = (
                    node.distance_to_end,
                    node.distance_to_start,
                )
            except AttributeError:
                pass
        self.start, self.end = self.end, self.start

    def find_topological_order(
        self, priority=True, apply_constraint=None, random=False
    ):
        """
        A list-schedule algorithm, it
        finds the topological order of the directed graph
        under certain constraint and priority indicator.
        The function returns a list of cycles,
        where each cycle is a list of instructions
        that can be executed in parallel.
        In the case of gates schedule,
        the result will be the gates cycle list.

        Parameters
        ----------
        priority: bool
            If use distance to the start and end nodes
            as a priority measure for the schedule problem.
        apply_constraint: function
            A Python function that determines
            if to instruction can be executed in parallel.
            E.g. if two gates apply to the same qubit, the function
            returns False.

        Returns
        -------
        cycles_list: list
            A list of cycles, where each cycle is a list of instructions
            that can be executed in parallel.
        constraint_dependency: set
            A set of instruction pairs that are found conflicted
            due to the hardware constraints.
            Because of this, they are executed in different cycles.
            This set is used to add this dependency to the graph
            in another method.
        """
        # The method will destruct the graph, therefore we make a copy.
        graph = deepcopy(self.nodes)
        cycles_list = []
        # available_gates: a list of available instructions,
        # all their dependence has been executed.
        available_gates = list(self.start)

        while available_gates:
            if random:
                shuffle(available_gates)
            if priority:
                available_gates.sort(key=cmp_to_key(self._compare_priority))
            current_cycle = []
            if apply_constraint is None:  # if no constraits
                current_cycle = deepcopy(available_gates)
            else:  # check if constraits allow the parallelization
                self._add_dependency_among_commuting_gates(
                    current_cycle, available_gates, apply_constraint
                )
            # add this cycle to cycles_list
            cycles_list.append(current_cycle)

            # update the list of available nodes
            # remove the executed nodes from available_node
            for node in current_cycle:
                available_gates.remove(node)
            # add new nodes to available_gates
            # if they have no other predecessors
            for node in current_cycle:
                for successor_ind in graph[node].successors:
                    graph[successor_ind].predecessors.remove(node)
                    if not graph[successor_ind].predecessors:
                        available_gates.append(successor_ind)
                graph[node].successors = set()

        return cycles_list

    def _add_dependency_among_commuting_gates(
        self, current_cycle, available_gates, apply_constraint
    ):
        for ind2 in available_gates:
            approval = True
            for ind1 in current_cycle:
                if not apply_constraint(ind2, ind1, self.nodes):
                    approval = False
                    # save the conflicted pairs of instructions
                    self.nodes[ind1].successors.add(ind2)
                    self.nodes[ind2].predecessors.add(ind1)
            if approval:
                current_cycle.append(ind2)

    def compute_distance(self, cycles_list):
        """
        Compute the longest distance of each node
        to the start and end nodes, including its own time.
        The weight for each dependency arrow is
        the duration of the source instruction
        (which should be 1 for gates schedule).
        The method solves the longest path problem
        by using the topological order in cycles_list.
        It makes sure that by following the list,
        the distance to the predecessors (successors) of
        the source (target) node is always calculated
        before the target (source) node.
        """
        cycles_list = cycles_list.copy()
        for node in self.nodes:
            node.distance_to_start = None
            node.distance_to_end = None
        self._compute_distance_to_start(cycles_list)
        self.reverse_graph()
        cycles_list.reverse()
        self._compute_distance_to_start(cycles_list)
        self.reverse_graph()

    def _compute_distance_to_start(self, cycles_list):
        """distance to the start node"""
        for ind in [ind for cycle in cycles_list for ind in cycle]:
            if not self.nodes[ind].predecessors:
                self.nodes[ind].distance_to_start = self.nodes[ind].duration
                continue
            self.nodes[ind].distance_to_start = (
                max(
                    [
                        self.nodes[p_ind].distance_to_start
                        for p_ind in self.nodes[ind].predecessors
                    ]
                )
                + self.nodes[ind].duration
            )

    def _compare_priority(self, ind1, ind2):
        """
        The node with longer `distance_to_end` has higher priority.
        If it is the same for the two nodes,
        the node with shorter `distance_to_start` has higher priority.
        If node1 has higher priority, the method returns a negative value.

        Parameters
        ----------
        ind1, ind2: int
            Indices of nodes.
        """
        return (
            # higher distance_to_end, higher priority
            self.nodes[ind2].distance_to_end
            - self.nodes[ind1].distance_to_end
        ) or (
            # if same distance_to_end, lower distance_to_start, higher priority
            self.nodes[ind1].distance_to_start
            - self.nodes[ind2].distance_to_start
        )


class Scheduler:
    """
    A gate (pulse) scheduler for quantum circuits (instructions).
    It schedules a given circuit or instructions
    to reduce the total execution time by parallelization.
    It uses heuristic methods mainly from
    in https://doi.org/10.1117/12.666419.

    The scheduler includes two methods,
    "ASAP", as soon as possible, and "ALAP", as late as possible.
    The later is commonly used in quantum computation
    because of the finite lifetime of qubits.

    The scheduler aims at pulse schedule and
    therefore does not consider merging gates to reduce the gates number.
    It assumes that the input circuit is optimized at the gate level
    and matches the hardware topology.

    Parameters
    ----------
    method: str
        "ASAP" for as soon as possible.
        "ALAP" for as late as possible.
    constraint_functions: list, optional
        A list of hardware constraint functions.
        Default includes a function `qubit_contraint`,
        i.e. one qubit cannot be used by two gates at the same time.
    """

    def __init__(
        self, method="ALAP", allow_permutation=True, constraint_functions=None
    ):
        self.method = method
        if constraint_functions is None:
            self.constraint_functions = [qubit_constraint]
        else:
            self.constraint_functions = constraint_functions
        self.allow_permutation = allow_permutation

    def schedule(
        self,
        circuit,
        gates_schedule=False,
        return_cycles_list=False,
        random_shuffle=False,
        repeat_num=0,
    ):
        """
        Schedule a `QubitCircuit`,
        a list of `Gates` or a list of `Instruction`.
        For pulse schedule, the execution time for each `Instruction`
        is given in its `duration` attributes.

        The scheduler first generates a quantum gates dependency graph,
        containing information about
        which gates have to be executed before some other gates.
        The graph preserves the mobility of the gates,
        i.e. commuting gates are not dependent on each other,
        even if they use the same qubits.
        Next, it computes the longest distance of each node
        to the start and end nodes.
        The distance for each dependency arrow is defined
        by the execution time of the instruction
        (By default, it is 1 for all gates).
        This is used as a priority measure in the next step.
        The gate with a longer distance to the end node and
        a shorter distance to the start node has higher priority.
        In the last step, it uses a list-schedule algorithm
        with hardware constraint and priority and
        returns a list of cycles for gates/instructions.

        For pulse schedule, an additional step is required
        to compute the start time of each instruction.
        It adds the additional dependency
        caused by hardware constraint to the graph
        and recomputes the distance of each node to the start and end node.
        This distance is then converted to
        the start time of each instruction.

        Parameters
        ----------
        circuit: QubitCircuit or list
            For gate schedule,
            it should be a QubitCircuit or a list of Gate objects.
            For pulse schedule, it should be a list of Instruction objects,
            each with an attribute `duration`
            that indicates the execution time of this instruction.
        gates_schedule: bool, optional
            `True`, if only gates schedule is needed.
            This saves some computation
            that is only useful to pulse schedule.
            If the input `circuit` is a `QubitCircuit`,
            it will be assigned to `True` automatically.
            Otherwise, the default is `False`.
        return_cycles_list: bool, optional
            If `True`, the method returns the `cycles_list`,
            e.g. [{0, 2}, {1, 3}],
            which means that the first cycle contains gates0 and gates2
            while the second cycle contains gates1 and gates3.
            It is only usefull for gates schedule.
        random_shuffle: bool, optional
            If the commuting gates are randomly scuffled to explore
            larger search space.
        repeat_num: int, optional
            Repeat the scheduling several times and use the best result.
            Used together with ``random_shuffle=True``.

        Returns
        -------
        gate_cycle_indices or instruction_start_time: list
            The cycle indices for each gate or
            the start time for each instruction.

        Examples
        --------
        >>> from qutip_qip.circuit import QubitCircuit
        >>> from qutip_qip.compiler import Scheduler
        >>> circuit = QubitCircuit(7)
        >>> circuit.add_gate("SNOT", 3)  # gate0
        >>> circuit.add_gate("CZ", 5, 3)  # gate1
        >>> circuit.add_gate("CZ", 4, 3)  # gate2
        >>> circuit.add_gate("CZ", 2, 3)  # gate3
        >>> circuit.add_gate("CZ", 6, 5)  # gate4
        >>> circuit.add_gate("CZ", 2, 6)  # gate5
        >>> circuit.add_gate("SWAP", [0, 2])  # gate6
        >>>
        >>> scheduler = Scheduler("ASAP")
        >>> scheduler.schedule(circuit, gates_schedule=True)
        [0, 1, 3, 2, 2, 3, 4]

        The result list is the cycle indices for each gate.
        It means that the circuit can be executed in 5 gate cycles:
        ``[gate0, gate1, (gate3, gate4), (gate2, gate5), gate6]``
        Notice that gate3 and gate4 commute with gate2,
        therefore, the order is changed to reduce the number of cycles.
        """
        circuit = deepcopy(circuit)
        if repeat_num > 0:
            random_shuffle = True
            result = [0]
            max_length = 4294967296
            for i in range(repeat_num):
                gate_cycle_indices = self.schedule(
                    circuit,
                    gates_schedule=gates_schedule,
                    return_cycles_list=return_cycles_list,
                    random_shuffle=random_shuffle,
                    repeat_num=0,
                )
                current_length = max(gate_cycle_indices)
                if current_length < max_length:
                    result = gate_cycle_indices
                    max_length = current_length
            return result

        if isinstance(circuit, QubitCircuit):
            gates = circuit.gates
        else:
            gates = circuit
        if not gates:
            return []

        # Generate the quantum operations dependency graph.
        instructions_graph = InstructionsGraph(gates)
        if self.allow_permutation:
            commutation_rules = self.commutation_rules
        else:
            commutation_rules = lambda *args, **kwargs: False
        instructions_graph.generate_dependency_graph(
            commuting=commutation_rules
        )
        if self.method == "ALAP":
            instructions_graph.reverse_graph()

        # Schedule without hardware constraints, then
        # use this cycles_list to compute the distance.
        cycles_list = instructions_graph.find_topological_order(
            priority=False, apply_constraint=None, random=random_shuffle
        )
        instructions_graph.compute_distance(cycles_list=cycles_list)

        # Schedule again with priority and hardware constraint.
        cycles_list = instructions_graph.find_topological_order(
            priority=True,
            apply_constraint=self.apply_constraint,
            random=random_shuffle,
        )

        # If we only need gates schedule, we can output the result here.
        if isinstance(gates[0], Gate):
            gates_schedule = True
        if gates_schedule or return_cycles_list:
            if self.method == "ALAP":
                cycles_list.reverse()
            if return_cycles_list:
                return cycles_list
            gate_cycles_indices = [0] * len(gates)
            for cycle_ind, cycle in enumerate(cycles_list):
                for instruction_ind in cycle:
                    gate_cycles_indices[instruction_ind] = cycle_ind
            return gate_cycles_indices

        # For pulse schedule,
        # we add the hardware dependency to the graph
        # and compute the longest distance to the start node again.
        # The longest distance to the start node determines
        # the start time of each pulse.
        instructions_graph.compute_distance(cycles_list=cycles_list)
        if self.method == "ALAP":
            instructions_graph.reverse_graph()

        # Output pulse schedule result.
        instruction_start_time = []
        for instruction in instructions_graph.nodes:
            instruction_start_time.append(
                instruction.distance_to_start - instruction.duration
            )
        return instruction_start_time

    def commutation_rules(self, ind1, ind2, instructions):
        """
        Determine if two gates commute, given that their used qubits overlap.
        Since usually the input gates are already in a universal gate sets,
        it uses an oversimplified condition:

        If the two gates do not have the same name,
        they are considered as not commuting.
        If they are the same gate and have the same controls or targets,
        they are considered as commuting.
        E.g. `CNOT 0, 1` commute with `CNOT 0, 2`.
        """
        instruction1 = instructions[ind1]
        instruction2 = instructions[ind2]
        if instruction1.name != instruction2.name:
            instruction1, instruction2 = sorted(
                [instruction1, instruction2],
                key=lambda instruction: instruction.name,
            )
            if instruction1.name == "CNOT" and instruction2.name in (
                "X",
                "RX",
            ):
                if instruction1.targets == instruction2.targets:
                    commute = True
                else:
                    commute = False
            elif instruction1.name == "CNOT" and instruction2.name in (
                "Z",
                "RZ",
            ):
                if instruction1.controls == instruction2.targets:
                    commute = True
                else:
                    commute = False
            else:
                commute = False
            return commute
        if (instruction1.controls) and (
            instruction1.controls == instruction2.controls
        ):
            commute = True
        elif instruction1.targets == instruction2.targets:
            commute = True
        else:
            commute = False
        return commute

    def apply_constraint(self, ind1, ind2, instructions):
        """
        Apply hardware constraint among the commuting gates to check
        if two instructions can be executed in parallel.

        Parameters
        ----------
        ind1, ind2: int
            indices of the two instructions
        instructions: list
            The instruction list
        """
        result = []
        for constraint_function in self.constraint_functions:
            result.append(constraint_function(ind1, ind2, instructions))
        return all(result)


def qubit_constraint(ind1, ind2, instructions):
    """
    Determine if two instructions have overlap in the used qubits.
    """
    if instructions[ind1].used_qubits & instructions[ind2].used_qubits:
        return False
    else:
        return True
