from io import TextIOWrapper
from tabulate import tabulate
from typing import Union
from logger import print
from numpy import linalg
import graphviz as gv
import string
from random import Random, shuffle
import time
from timer import Timer


# def char_map(x): return char_map(x // 26) + string.ascii_lowercase[x % 26] if x // 26 else string.ascii_lowercase[x]
def char_map(x): return x


class BadFormat(Exception):
    pass


class Matrix():

    def __init__(self, rows: int, cols: int, fill: Union[int, float] = 0):
        self.rows_size = rows
        self.cols_size = cols
        self._rows = None
        self._cols = None
        self.matrix = [[fill for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, index: tuple[int, int]):
        return self.matrix[index[0]][index[1]]

    def __setitem__(self, index: tuple[int, int], value: Union[int, float]):
        self.matrix[index[0]][index[1]] = value

    def __str__(self):
        return tabulate(self.matrix, tablefmt="fancy_grid")

    def __repr__(self) -> str:
        return f"Matrix({self.rows_size}, {self.cols_size})"

    def __iter__(self):
        for cell in self.flatten():
            yield cell

    def flatten(self):
        return [cell for row in self.matrix for cell in row]

    def determinant(self):
        if self.rows_size != self.cols_size:
            raise ValueError("The matrix must be square")
        return linalg.det(self.matrix)

    @property
    def rows(self) -> list[list[Union[int, float]]]:
        return [row for row in self.matrix]

    @property
    def cols(self) -> list[list[Union[int, float]]]:
        return [[row[i] for row in self.matrix] for i in range(self.cols_size)]

    def __len__(self):
        return self.cols_size * self.rows_size

    @property
    def dimension(self):
        return (self.rows_size, self.cols_size)

    @property
    def length(self):
        return len(self)

    @staticmethod
    def from_list(lst) -> 'Matrix':
        # check if the list is have the size for each row
        if len(set(map(len, lst))) != 1:
            raise BadFormat("The list is not a matrix")
        rows = len(lst)
        cols = len(lst[0])
        matrix = Matrix(rows, cols)
        for i, row in enumerate(lst):
            for j, cell in enumerate(row):
                matrix[(i, j)] = cell
        return matrix

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.dimension != other.dimension:
            raise ValueError("The two matrices must have the same dimension")
        matrix = Matrix(self.rows_size, self.cols_size)
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                matrix[(i, j)] = cell + other[(i, j)]
        return matrix

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.dimension != other.dimension:
            raise ValueError("The two matrices must have the same dimension")
        matrix = Matrix(self.rows_size, self.cols_size)
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                matrix[(i, j)] = cell - other[(i, j)]
        return matrix

    def __mul__(self, other: Union[int, float, 'Matrix']) -> 'Matrix':
        if isinstance(other, (int, float)):
            matrix = Matrix(self.rows_size, self.cols_size)
            for i, row in enumerate(self.matrix):
                for j, cell in enumerate(row):
                    matrix[(i, j)] = cell * other
            return matrix
        elif isinstance(other, Matrix):
            if self.cols_size != other.rows_size:
                raise ValueError("The two matrices cannot be multiplied")
            matrix = Matrix(self.rows_size, other.cols_size)
            for i, row in enumerate(self.matrix):
                for j, cell in enumerate(other.cols()):
                    matrix[(i, j)] = sum([cell1 * cell2 for cell1, cell2 in zip(row, cell)])
            return matrix
        else:
            raise ValueError("The other matrix must be a scalar or a matrix")

    def __rmul__(self, other: Union[int, float, 'Matrix']) -> 'Matrix':
        return self.__mul__(other)

    def __imul__(self, other: Union[int, float, 'Matrix']) -> 'Matrix':
        return self.__mul__(other)

    def __iadd__(self, other: 'Matrix') -> 'Matrix':
        return self.__add__(other)

    def __isub__(self, other: 'Matrix') -> 'Matrix':
        return self.__sub__(other)

    def __eq__(self, other: 'Matrix') -> bool:
        return self.matrix == other.matrix

    def __ne__(self, other: 'Matrix') -> bool:
        return not self.__eq__(other)

    def __contains__(self, item: Union[int, float]) -> bool:
        return item in self.flatten()

    def __min__(self) -> Union[int, float]:
        return min(self.flatten())

    def __max__(self) -> Union[int, float]:
        return max(self.flatten())

    def index(self, item: Union[int, float]) -> tuple[int, int]:
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                if cell == item:
                    return (i, j)
        raise ValueError("Item not found")

    def copy(self) -> 'Matrix':
        matrix = Matrix(self.rows_size, self.cols_size)
        matrix.matrix = [row.copy() for row in self.matrix]
        return matrix


class TransportationTable:

    missing_edge_buffer = None

    seed = time.time()  # ? this is used to get the same random values and the same results each time

    @property
    def random(self) -> Random:
        return Random(self.seed)

    def __init__(self) -> None:
        self.supply = []
        self.demand = []
        self.costs: Matrix = Matrix(0, 0)
        self.transportation_table = Matrix(0, 0)

    @property
    @Timer.timeit
    def marginal_costs(self) -> Matrix:
        s, t = self.potentials()
        potentials = Matrix(self.costs.rows_size, self.costs.cols_size)
        for i in range(self.costs.rows_size):
            for j in range(self.costs.cols_size):
                potentials[(i, j)] = s[i] - t[j]
        return self.costs - potentials

    @property
    def potential_costs(self) -> Matrix:
        s, t = self.potentials()
        potentials = Matrix(self.costs.rows_size, self.costs.cols_size)
        for i in range(self.costs.rows_size):
            for j in range(self.costs.cols_size):
                potentials[(i, j)] = s[i] - t[j]
        return potentials

    @staticmethod
    def from_file(file: TextIOWrapper) -> 'TransportationTable':
        table = TransportationTable()
        # first line is the dimension of the matrix
        rows, cols = map(int, file.readline().split(" "))
        table.supply = [0] * rows
        table.demand = [0] * cols
        matrix = Matrix(rows, cols)
        for i in range(rows):
            # last column is the supply
            row = list(map(int, file.readline().split(" ")))
            table.supply[i] = row.pop()
            if len(row) != cols:
                raise BadFormat("The matrix is not well formatted")
            for j, cell in enumerate(row):
                matrix[(i, j)] = cell
        # last line is the demand
        table.demand = list(map(int, file.readline().split(" ")))
        if len(table.demand) != cols:
            raise BadFormat("The demand is not well formatted")
        if sum(table.supply) != sum(table.demand):
            raise ValueError("The supply and demand are not equal")
        table.costs = matrix
        return table

    def __str__(self) -> str:
        table = []
        for i, row in enumerate(self.costs.rows):
            table.append(row + [self.supply[i]])
        # add the demand
        table.append(self.demand)
        # headers
        headers = [f"C_{char_map(i)}" for i in range(len(self.supply))] + ["Provitions"]
        # index
        index = [f"S_{i + 1}" for i in range(len(self.demand))] + ["Orders"]
        return tabulate(table, tablefmt="fancy_grid", showindex=index, headers=headers)

    def __repr__(self) -> str:
        return f"TransportationTable({len(self.supply)}, {len(self.demand)})"

    def display(self) -> None:
        print(self)

    def show(self, matrix: Matrix, rows: list = None, cols: list = None, row_name: str = "Provitions", col_name: str = "Orders") -> None:
        if rows is None:
            rows = self.supply
        if cols is None:
            cols = self.demand
        table = []
        for i, row in enumerate(matrix.rows):
            table.append(row + [rows[i]])
        # add the demand
        table.append(cols)
        # headers
        headers = [f"C_{char_map(i)}" for i in range(len(self.demand))] + [row_name]
        # index
        index = [f"S_{i + 1}" for i in range(len(self.supply))] + [col_name]
        print(tabulate(table, tablefmt="fancy_grid", showindex=index, headers=headers))

    @Timer.timeit
    def NorthWestCorner(self) -> None:
        # create a matrix with the same dimension
        matrix = Matrix(len(self.supply), len(self.demand))
        supply, demand = self.supply.copy(), self.demand.copy()
        i = 0
        j = 0
        while i < len(supply) and j < len(demand):
            # get the minimum between the supply and demand
            q = min(supply[i], demand[j])
            matrix[(i, j)] = q
            supply[i] -= q
            demand[j] -= q
            if supply[i] == 0:
                i += 1
            if demand[j] == 0:
                j += 1
        self.transportation_table = matrix

    def _get_edges(self) -> list[tuple[int, int]]:
        # list the indexes where the values are not null in the transportation table
        indexes: list[tuple[int, int]] = []
        for i, row in enumerate(self.transportation_table.rows):
            for j, cell in enumerate(row):
                if cell != 0:
                    indexes.append((i, j))
        return indexes

    #! this function is not working properly
    def _missing_edges_depreciated(self, indexes: list[tuple[int, int]]) -> None:
        missing_edges = []
        indexes = indexes.copy()
        for j in indexes.copy():
            # check if the cell is alone in its row and column regarding the other cells
            if sum([1 for index in indexes if index[0] == j[0]]) == 1 and sum([1 for index in indexes if index[1] == j[1]]) == 1:
                # if the cell is alone in its row and column then link it to the smallest cost cell in the same row or column which is not the cell itself
                print("Cell alone : ", j)
                row = self.costs.rows[j[0]]
                col = self.costs.cols[j[1]]
                print("Row : ", row)
                print("Col : ", col)
                val = max(row + col)
                for k, (rcell, ccell) in enumerate(zip(row, col)):
                    #! sometime we can have two equivalent cells with the same cost
                    if k != j[1] and k != j[0] and (rcell == val or ccell == val):
                        print("Equivalent cells", rcell, ccell, val, (k, j[1]), (j[0], k))
                    if k != j[1] and ccell < val:
                        val = ccell
                        new_index = (j[0], k)
                    if k != j[0] and rcell < val:
                        val = rcell
                        new_index = (k, j[1])
                if new_index not in indexes:
                    indexes.append(new_index)
                    missing_edges.append(new_index)
                else:
                    raise ValueError("Error in the new indexes allocation")
        return missing_edges

    @Timer.timeit
    def _missing_edges_slow(self, indexes: list[tuple[int, int]]) -> None:
        missing_edges = []
        indexes = indexes.copy()
        nb_missing_edges = len(self.supply) + len(self.demand) - len(indexes) - 1
        if self.missing_edge_buffer is not None and nb_missing_edges > 0:
            missing_edges.append(self.missing_edge_buffer)
            indexes.append(self.missing_edge_buffer)
        gn = self.random  # ? get the actual random generator
        for i in range(nb_missing_edges):
            # find the edge with the minimum cost
            edges = [(i, j) for i in range(len(self.supply)) for j in range(len(self.demand)) if (i, j) not in indexes]
            # if there is some equivalent costs then randomize the order
            sorted_edges = sorted(edges, key=lambda x: (self.costs[x], gn.random()))
            # check if the edge can be added without creating a cycle
            for i in range(len(sorted_edges)):
                graph = Graph.from_list_index(indexes + [sorted_edges[i]])
                if not graph.has_cycle():
                    indexes.append(sorted_edges[i])
                    missing_edges.append(sorted_edges[i])
                    break
        # check if the number of missing edges is correct
        if len(missing_edges) != nb_missing_edges:
            raise ValueError("Error in the missing edges allocation")
        return missing_edges

    @Timer.timeit
    def _missing_edges(self, indexes: list[tuple[int, int]]) -> None:
        missing_edges = []
        indexes = set(indexes)  # Convert to set for faster membership check
        supply_length = len(self.supply)
        demand_length = len(self.demand)
        nb_missing_edges = supply_length + demand_length - len(indexes) - 1

        if self.missing_edge_buffer is not None and nb_missing_edges > 0:
            missing_edges.append(self.missing_edge_buffer)
            indexes.add(self.missing_edge_buffer)

        # Initialize random generator with the class seed
        gn = self.random
        all_edges = {(i, j) for i in range(supply_length) for j in range(demand_length)}

        # Shuffle the edges randomly
        shuffled_edges = list(all_edges - indexes)
        gn.shuffle(shuffled_edges)

        # Sort the shuffled edges by cost
        sorted_edges = sorted(shuffled_edges, key=lambda x: self.costs[x])

        for edge in sorted_edges:
            if len(missing_edges) == nb_missing_edges:
                break
            graph = Graph.from_list_index(list(indexes) + [edge])
            if not graph.has_cycle():
                indexes.add(edge)
                missing_edges.append(edge)

        if len(missing_edges) != nb_missing_edges:
            raise ValueError("Error in the missing edges allocation")

        return missing_edges

    @Timer.timeit
    def potentials(self) -> tuple[list[int], list[int]]:
        size = len(self.supply) + len(self.demand)
        # potentials are binds by the following equation
        #  c_ij = s_i - t_j
        s = [None] * len(self.supply)
        t = [None] * len(self.demand)
        # make the system of equations matrix to solve
        matrix = Matrix(size, size)
        indexes = self._get_edges()
        missing = self._missing_edges(indexes)
        indexes += missing
        g = Graph.from_list_index(indexes)
        # g.display()
        # fill the matrix
        for i, index in enumerate(indexes):
            matrix[(i, index[0])] = 1
            matrix[(i, len(self.supply) + index[1])] = -1
        else:
            # init one of the potentials to 0
            matrix[(size - 1, 0)] = 1
            i = 0
            while matrix.determinant() == 0 and i < size - 1:
                matrix[(size - 1, i)] = 0
                i += 1
                matrix[(size - 1, i)] = 1
        # create the vector of costs
        costs = [self.costs[index] for index in indexes]
        costs.append(0)
        try:
            res = linalg.solve(matrix.matrix, costs)
        except linalg.LinAlgError:
            g = Graph.from_list_index(indexes)
            g.display()
            print("Indexes : ", indexes)
            print("nb indexes : ", len(indexes))
            print("Missing : ", missing)
            print("nb missing : ", len(missing))
            print("Matrix det: ", matrix.determinant())
            print("Doublon in indexes : ", len(set(indexes)) != len(indexes))
            print("Graph is degenerate : ", g.is_degenerate())
            print("Actual transport table : ")
            self.show(self.transportation_table)
            raise ValueError("The system of equations is not solvable")
        # assign the values to the potentials
        s = [*map(int, res[:len(self.supply)])]
        t = [*map(int, res[len(self.supply):])]
        return (s, t)

    @Timer.timeit
    def BalasHammer(self) -> None:
        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy().matrix
        allocations = Matrix(self.costs.rows_size, self.costs.cols_size)
        while sum(supply) > 0 and sum(demand) > 0:
            penalties = self.penalties(supply, demand, costs)
            edge = self.choose_edge_to_fill(penalties, costs, allocations, supply, demand)
            if edge is None:
                break
            self.fill_edge(allocations, edge, supply, demand, costs)
        self.transportation_table = allocations

    def penalties(self, supply, demand, costs) -> list[tuple[int, int, str]]:
        penalties = []
        for i in range(len(supply)):
            if supply[i] > 0:
                row_costs = costs[i][:]
                row_costs.sort()
                penalty = row_costs[1] - row_costs[0]
                if penalty != float('inf'):
                    penalties.append(((i), penalty, "row"))
        for i in range(len(demand)):
            if demand[i] > 0:
                column_costs = [costs[x][i] for x in range(len(costs))]
                column_costs.sort()
                penalty = column_costs[1] - column_costs[0]
                if penalty != float('inf'):
                    penalties.append(((i), penalty, "column"))
        penalties.sort(key=lambda x: x[1], reverse=True)
        return penalties

    def choose_edge_to_fill(self, penalties, costs, allocations, supply, demand) -> Union[tuple[int, int], None]:
        for cell, penalty, penalty_type in penalties:
            i = cell
            if penalty_type == "row":
                min_cost = float('inf')
                min_cost_index = None
                for j, cost in enumerate(costs[i]):
                    if supply[i] > 0 and demand[j] > 0 and allocations[(i, j)] == 0:
                        if cost < min_cost:
                            min_cost = cost
                            min_cost_index = j
                if min_cost_index is not None:
                    #!print(f"{penalty_type}: {cell}, Penalty: {penalty}")
                    return (i, min_cost_index)
            elif penalty_type == "column":
                min_cost = float('inf')
                min_cost_index = None
                for k, row in enumerate(costs):
                    if supply[k] > 0 and demand[i] > 0 and allocations[(k, i)] == 0:
                        if row[i] < min_cost:
                            min_cost = row[i]
                            min_cost_index = k
                if min_cost_index is not None:
                    #!print(f"{penalty_type}: {cell}, Penalty: {penalty}")
                    return (min_cost_index, i)
        return None

    def fill_edge(self, allocations, edge, supply, demand, costs) -> None:
        i, j = edge
        allocation = min(supply[i], demand[j])
        allocations[(i, j)] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        if supply[i] == 0:
            for k in range(len(costs[i])):
                costs[i][k] = float('inf')

    def get_graph(self, fill=False) -> 'Graph':
        graph = Graph()
        indexes = self._get_edges()
        if fill:
            indexes += self._missing_edges(indexes)
        for index in indexes:
            p = State(f"S_{index[0] + 1}", self.supply[index[0]])
            o = State(f"C_{char_map(index[1])}", self.demand[index[1]])
            graph.states.add(p)
            graph.states.add(o)
            graph.edges.append(Edge(p, o, self.transportation_table[index], index))
        return graph

    @Timer.timeit
    def optimize(self) -> bool:
        # check if there is a cycle in the actual proposition
        first = True
        nb_iter = 0
        while True:
            print("Iteration", nb_iter)
            nb_iter += 1
            # Step 1 : Randomize the seed to get a different result each iteration and avoid getting stuck in a loop
            self.seed = time.time()
            # Step 2 : Check if the graph has a cycle and remove it
            graph = self.get_graph()
            cycle = graph.has_cycle()
            if cycle:
                self.stepping_stone(cycle)
                graph = self.get_graph()
            # Step 3 : Compute the marginal costs
            marginal_costs = self.marginal_costs
            min_mcost = min(marginal_costs)
            # Step 4 : Check if the table is optimized or not with the minimum marginal cost
            if min_mcost >= 0:
                if first:
                    #!print("The transportation table is already optimized")
                    return False
                # if the minimum marginal cost is positive or null then we can't optimize the table anymore
                break
            # Step 5 : Add the edge with the minimum marginal cost to the graph and use the stepping stone method
            index = marginal_costs.index(min_mcost)
            # get the graph
            graph = self.get_graph(fill=graph.is_degenerate())  # ! this line sometimes doesn'nt get the same missing edges
            # add the edge to the graph
            state = State(f"S_{index[0] + 1}", self.supply[index[0]])
            next_state = State(f"C_{char_map(index[1])}", self.demand[index[1]])
            graph.edges.append(Edge(state, next_state, 0, index))
            # find the added cycle
            cycle: list[Edge] = graph.has_cycle()
            #!print("Cycle : ", end='')
            #!print(*cycle, sep=' -> ')
            # optimize the table
            delta = self.stepping_stone(cycle)
            if delta == 0:
                #!print("The cost didn't change")
                self.missing_edge_buffer = index
            else:
                self.missing_edge_buffer = None
            #!print(f"Delta : {delta}")
            #!self.show(self.transportation_table)
            # input("Press a key...")
            first = False
        return True

    @property
    def total_cost(self) -> int:
        return sum([self.costs[index] * self.transportation_table[index] for index in self._get_edges()])

    def stepping_stone(self, cycle: list['Edge']) -> Union[int, float]:
        delta_cost = [self.costs[edge.matrix_index] * (-1)**i for i, edge in enumerate(cycle)]
        delta_cost = sum(delta_cost)
        if delta_cost > 0:
            delta = -min([self.transportation_table[edge.matrix_index] for edge in cycle if cycle.index(edge) % 2 == 0])
        elif delta_cost < 0:
            delta = min([self.transportation_table[edge.matrix_index] for edge in cycle if cycle.index(edge) % 2 == 1])
        else:
            raise ValueError("The cost didn't change")
        if delta == 0:
            return 0
        for i, edge in enumerate(cycle):
            if i % 2 == 0:
                self.transportation_table[edge.matrix_index] += delta
            else:
                self.transportation_table[edge.matrix_index] -= delta
        return delta

    @Timer.timeit
    def NordWestOptimized(self):
        self.NorthWestCorner()
        self.optimize()

    @Timer.timeit
    def BalasHammerOptimized(self):
        self.BalasHammer()
        print("BalasHammer")
        self.optimize()


class Graph:

    def __init__(self) -> None:
        self.states: set[State] = set()
        self.edges: list[Edge] = []

    def __str__(self) -> str:
        return f"States : {self.states}\nEdges : {[f'{state.name} -> {next_state.name}' for state, next_state, _ in self.edges]}"

    def __repr__(self) -> str:
        return f"Graph {id(self)}"

    def display(self, cycle: list['Edge'] = []) -> None:
        """
        Display the graph using graphviz
        """
        graph = gv.Digraph()
        for state in self.states:
            graph.node(state.name, label=str(state))
        for edge in self.edges:
            color = "black"
            state, next_state, value = edge
            if edge in cycle or edge.reversed() in cycle:
                color = "red"
            graph.edge(state.name, next_state.name, label=str(value), arrowhead="none", color=color)
        graph.view(cleanup=True)
        input("Press a key...")

    def is_degenerate(self) -> bool:
        return len(self.edges) < len(self.states) - 1 or bool(self.has_cycle())

    @Timer.timeit
    def has_cycle(self) -> Union[list['Edge'], bool]:
        # Check if there is a cycle in the graph
        # edges aren't directed when checking for cycles
        visited = set()
        recursion_stack = list()

        def dfs(state, parent):
            visited.add(state)
            recursion_stack.append(state)
            # ? get the edges linked to the current state and the reversed edges
            edges = [edge for edge in self.edges if edge[0] == state]
            edges += [edge.reversed() for edge in self.edges if edge[1] == state]
            for _, next_state, _ in edges:
                # check if the next state is the parent of the current state and we are just going back with previous edge
                # ? only work since a state can't be linked to itself in any case
                if parent is not None and next_state == parent:
                    continue
                # check if the next state has already been visited, if not visit it
                if next_state not in visited:
                    path = dfs(next_state, state)
                    if path:
                        return path
                # check if the next state is in the recursion stack
                elif next_state in recursion_stack:
                    # clear the beginning of the recursion stack until the first occurence of the next state so we get the cycle
                    while recursion_stack[0] != next_state and len(recursion_stack) > 0:
                        recursion_stack.pop(0)
                    # translate the list of states to a list of edges
                    cycle = []
                    possible_edge = self.edges + [edge.reversed() for edge in self.edges]
                    # ? get original edges and not generated ones to retrieve the indexes
                    for i in range(len(recursion_stack)):
                        # get the edge between the current state and the next state in the recursion stack
                        for edge in possible_edge:
                            if edge[0] == recursion_stack[i] and edge[1] == recursion_stack[(i + 1) % len(recursion_stack)]:
                                cycle.append(edge)
                                break
                    return cycle
            recursion_stack.remove(state)
            return False

        for state in self.states:
            if state not in visited:
                cycle = dfs(state, None)
                if cycle:
                    return cycle
        return False

    @ staticmethod
    def from_list_index(lst: list[tuple[int, int]]) -> 'Graph':
        graph = Graph()
        for index in lst:
            state = State(f"S_{index[0] + 1}", 0)
            next_state = State(f"C_{char_map(index[1])}", 0)
            graph.states.add(state)
            graph.states.add(next_state)
            graph.edges.append(Edge(state, next_state, 0, index))
        return graph


class State:

    def __init__(self, name: str, weight: int) -> None:
        self.name = name
        self.weight = weight

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"State({self.name})"

    def __eq__(self, other: 'State') -> bool:
        return self.name == other.name and self.weight == other.weight

    def __ne__(self, other: 'State') -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.name, self.weight))


class Edge(tuple[State, State, int]):
    def __init__(self, state: State, next_state: State, value: int, index: tuple[int, int]) -> None:
        super().__init__()
        self.state = state
        self.next_state = next_state
        self.value = value
        self.matrix_index = index

    def __new__(cls, state: State, next_state: State, value: int, index: tuple[int, int]):
        return super().__new__(cls, (state, next_state, value))

    def __str__(self) -> str:
        return f"({self.state},{self.next_state})"

    def __repr__(self) -> str:
        return f"Edge({self.state}, {self.next_state}, {self.value})"

    def reversed(self) -> 'Edge':
        return Edge(self.next_state, self.state, self.value, self.matrix_index)

    def __eq__(self, value: object) -> bool:
        return self.state == value.state and self.next_state == value.next_state and self.value == value.value

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return hash((self.state, self.next_state, self.value))


if __name__ == "__main__":

    def test_transportation_table(i):
        print(f"Test {i}")
        with open(f"data/test{i}.txt", "r") as f:
            table = TransportationTable.from_file(f)
        print("Costs matrix :")
        table.show(table.costs)
        print("Balas algorithm :")
        table.NorthWestCorner()
        table.show(table.transportation_table)
        print("Total cost : ", table.total_cost)
        row, col = table.potentials()
        print("Marginal costs and potentials :")
        table.show(table.marginal_costs, row, col, "potentials", "potentials")
        # table.get_graph().display()
        print("Optimizing the transportation table...")
        if table.optimize():
            print("Optimized transportation table :")
            table.show(table.transportation_table)
            print(f"Total cost : {table.total_cost}")

    for i in range(100):
        test_transportation_table(2)
