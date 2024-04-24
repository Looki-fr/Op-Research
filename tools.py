from hmac import new
from io import TextIOWrapper
from math import e
from operator import ne
from tabulate import tabulate
from typing import Union
from logger import print
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from numpy import linalg, rec
import graphviz as gv


class BadFormat(Exception):
    pass


class Index:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __iadd__(self, other):
        # if other is a tuple
        if isinstance(other, tuple):
            self.row += other[0]
            self.col += other[1]
        # if other is an Index
        elif isinstance(other, Index):
            self.row += other.row
            self.col += other.col
        if self.row < 0 or self.col < 0:
            raise IndexError("Index out of bounds")
        return self

    def __add__(self, other):
        return self.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, tuple):
            self.row -= other[0]
            self.col -= other[1]
        elif isinstance(other, Index):
            self.row -= other.row
            self.col -= other.col
        if self.row < 0 or self.col < 0:
            raise IndexError("Index out of bounds")
        return self

    def __sub__(self, other):
        return self.__isub__(other)

    def __mul__(self, other):
        return Index(self.row * other, self.col * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __str__(self):
        return f"({self.row}, {self.col})"

    def __repr__(self) -> str:
        return f"Index({self.row}, {self.col})"


class Matrix():

    def __init__(self, rows: int, cols: int, fill: Union[int, float] = 0):
        self.rows_size = rows
        self.cols_size = cols
        self._rows = None
        self._cols = None
        self.matrix = [[fill for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, index: Index):
        return self.matrix[index.row][index.col]

    def __setitem__(self, index: Index, value: Union[int, float]):
        self.matrix[index.row][index.col] = value

    def __str__(self):
        return tabulate(self.matrix, tablefmt="fancy_grid")

    def __repr__(self) -> str:
        return f"Matrix({self.rows_size}, {self.cols_size})"

    def __iter__(self):
        for cell in self.flatten():
            yield cell

    def flatten(self):
        return [cell for row in self.matrix for cell in row]

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
                matrix[Index(i, j)] = cell
        return matrix

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.dimension != other.dimension:
            raise ValueError("The two matrices must have the same dimension")
        matrix = Matrix(self.rows_size, self.cols_size)
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                matrix[Index(i, j)] = cell + other[Index(i, j)]
        return matrix

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.dimension != other.dimension:
            raise ValueError("The two matrices must have the same dimension")
        matrix = Matrix(self.rows_size, self.cols_size)
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                matrix[Index(i, j)] = cell - other[Index(i, j)]
        return matrix

    def __mul__(self, other: Union[int, float, 'Matrix']) -> 'Matrix':
        if isinstance(other, (int, float)):
            matrix = Matrix(self.rows_size, self.cols_size)
            for i, row in enumerate(self.matrix):
                for j, cell in enumerate(row):
                    matrix[Index(i, j)] = cell * other
            return matrix
        elif isinstance(other, Matrix):
            if self.cols_size != other.rows_size:
                raise ValueError("The two matrices cannot be multiplied")
            matrix = Matrix(self.rows_size, other.cols_size)
            for i, row in enumerate(self.matrix):
                for j, cell in enumerate(other.cols()):
                    matrix[Index(i, j)] = sum([cell1 * cell2 for cell1, cell2 in zip(row, cell)])
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

    def index(self, item: Union[int, float]) -> Index:
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                if cell == item:
                    return Index(i, j)
        raise ValueError("Item not found")

    def copy(self) -> 'Matrix':
        matrix = Matrix(self.rows_size, self.cols_size)
        matrix.matrix = [row.copy() for row in self.matrix]
        return matrix


class TransportationTable:

    def __init__(self) -> None:
        self.supply = []
        self.demand = []
        self.costs: Matrix = Matrix(0, 0)
        self.transportation_table = Matrix(0, 0)

    @property
    def marginal_costs(self) -> Matrix:
        s, t = self.potentials()
        potentials = Matrix(self.costs.rows_size, self.costs.cols_size)
        for i in range(self.costs.rows_size):
            for j in range(self.costs.cols_size):
                potentials[Index(i, j)] = s[i] - t[j]
        return self.costs - potentials

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
                matrix[Index(i, j)] = cell
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
        headers = [f"C_{i}" for i in range(len(self.supply))] + ["Provitions"]
        # index
        index = [f"P_{i}" for i in range(len(self.demand))] + ["Orders"]
        return tabulate(table, tablefmt="fancy_grid", showindex=index, headers=headers)

    def __repr__(self) -> str:
        return f"TransportationTable({len(self.supply)}, {len(self.demand)})"

    def display(self) -> None:
        print(self)

    def show(self, matrix: Matrix, rows: list = None, cols: list = None) -> None:
        if rows is None:
            rows = self.supply
        if cols is None:
            cols = self.demand
        table = []
        for i, row in enumerate(matrix.rows):
            table.append(row + [rows[i]])
        # add the demand
        table.append(cols)
        print(table)
        # headers
        headers = [f"C_{i}" for i in range(len(self.demand))] + ["Provitions"]
        # index
        index = [f"P_{i}" for i in range(len(self.supply))] + ["Orders"]
        print(tabulate(table, tablefmt="fancy_grid", showindex=index, headers=headers))

    def NorthWestCorner(self) -> None:
        # create a matrix with the same dimension
        matrix = Matrix(len(self.supply), len(self.demand))
        supply, demand = self.supply.copy(), self.demand.copy()
        i = 0
        j = 0
        while i < len(supply) and j < len(demand):
            # get the minimum between the supply and demand
            q = min(supply[i], demand[j])
            matrix[Index(i, j)] = q
            supply[i] -= q
            demand[j] -= q
            if supply[i] == 0:
                i += 1
            if demand[j] == 0:
                j += 1
        self.transportation_table = matrix

    def get_transportation_indexes(self) -> list[Index]:
        # list the indexes where the values are not null in the transportation table
        indexes: list[Index] = []
        for i, row in enumerate(self.transportation_table.rows):
            for j, cell in enumerate(row):
                if cell != 0:
                    indexes.append(Index(i, j))
        else:
            # if the number of indexes is less than the (size of the matrix - 1) then we must add some constraints
            # to do that we must link cell which are alone in both their row and column
            for j in indexes.copy():
                # check if the cell is alone in its row and column regarding the other cells
                if sum([1 for index in indexes if index.row == j.row]) == 1 and sum([1 for index in indexes if index.col == j.col]) == 1:
                    # if the cell is alone in its row and column then link it to the smallest cost cell in the same row or column which is not the cell itself
                    row = self.costs.rows[j.row]
                    col = self.costs.cols[j.col]
                    val = max(row + col)
                    for k, (rcell, ccell) in enumerate(zip(row, col)):
                        if k != j.col and ccell < val:
                            val = ccell
                            new_index = Index(j.row, k)
                        if k != j.row and rcell < val:
                            val = rcell
                            new_index = Index(k, j.col)
                    if new_index not in indexes:
                        indexes.append(new_index)
                    else:
                        raise ValueError("Error in the new indexes allocation")
        return indexes

    def potentials(self) -> tuple[list[int], list[int]]:
        size = len(self.supply) + len(self.demand)
        # potentials are binds by the following equation
        #  c_ij = s_i - t_j
        s = [None] * len(self.supply)
        t = [None] * len(self.demand)
        # make the system of equations matrix to solve
        matrix = Matrix(size, size)
        indexes = self.get_transportation_indexes()
        # fill the matrix
        for i, index in enumerate(indexes):
            matrix[Index(i, index.row)] = 1
            matrix[Index(i, len(self.supply) + index.col)] = -1
        else:
            # init one of the potentials to 0
            matrix[Index(size - 1, 0)] = 1
        # create the vector of costs
        costs = [self.costs[index] for index in indexes]
        costs.append(0)
        res = linalg.solve(matrix.matrix, costs)
        # assign the values to the potentials
        s = [*map(int, res[:len(self.supply)])]
        t = [*map(int, res[len(self.supply):])]
        return (s, t)

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
                    if supply[i] > 0 and demand[j] > 0 and allocations[Index(i, j)] == 0:
                        if cost < min_cost:
                            min_cost = cost
                            min_cost_index = j
                if min_cost_index is not None:
                    print(f"{penalty_type}: {cell}, Penalty: {penalty}")
                    return (i, min_cost_index)
            elif penalty_type == "column":
                min_cost = float('inf')
                min_cost_index = None
                for k, row in enumerate(costs):
                    if supply[k] > 0 and demand[i] > 0 and allocations[Index(k, i)] == 0:
                        if row[i] < min_cost:
                            min_cost = row[i]
                            min_cost_index = k
                if min_cost_index is not None:
                    print(f"{penalty_type}: {cell}, Penalty: {penalty}")
                    return (min_cost_index, i)
        return None

    def fill_edge(self, allocations, edge, supply, demand, costs) -> None:
        i, j = edge
        allocation = min(supply[i], demand[j])
        allocations[Index(i, j)] = allocation
        supply[i] -= allocation
        demand[j] -= allocation
        if supply[i] == 0:
            for k in range(len(costs[i])):
                costs[i][k] = float('inf')

    def get_graph(self) -> 'Graph':
        graph = Graph()
        indexes = self.get_transportation_indexes()
        for index in indexes:
            p = State(f"P_{index.row}", self.supply[index.row])
            o = State(f"O_{index.col}", self.demand[index.col])
            graph.states.add(p)
            graph.states.add(o)
            graph.edges.append((p, o, self.transportation_table[index]))
        return graph


class Graph:

    def __init__(self) -> None:
        self.states: set[State] = set()
        self.edges: list[tuple[State, State, int]] = []

    def __str__(self) -> str:
        return f"States : {self.states}\nEdges : {[f'{state.name} -> {next_state.name}' for state, next_state, _ in self.edges]}"

    def __repr__(self) -> str:
        return f"Graph {id(self)}"

    def display(self) -> None:
        """
        Display the graph using graphviz
        """
        graph = gv.Digraph()
        for state in self.states:
            graph.node(state.name, label=str(state))
        for state, next_state, value in self.edges:
            graph.edge(state.name, next_state.name, label=str(value))
        graph.view(cleanup=True)

    def is_degenerate(self) -> bool:
        return len(self.edges) != len(self.states) - 1 or self.has_cycle()

    def has_cycle(self) -> Union[list['State'], bool]:
        # Check if there is a cycle in the graph
        # edges aren't directed when checking for cycles
        visited = set()
        recursion_stack = list()

        def dfs(state, parent):
            visited.add(state)
            recursion_stack.append(state)

            edges = [edge for edge in self.edges if edge[0] == state]
            edges += [(edge[1], edge[0], edge[2]) for edge in self.edges if edge[1] == state]
            for _, next_state, _ in edges:
                if next_state == state or parent is not None and next_state == parent:
                    continue
                if next_state not in visited:
                    path = dfs(next_state, state)
                    if path:
                        return path
                elif next_state in recursion_stack:
                    # clear the beginning of the recursion stack until the first occurence of the next state
                    while recursion_stack[0] != next_state and len(recursion_stack) > 0:
                        recursion_stack.pop(0)
                    return recursion_stack
            recursion_stack.remove(state)
            return False

        for state in self.states:
            if state not in visited:
                cycle = dfs(state, None)
                if cycle:
                    return cycle
        return False


class State:

    def __init__(self, name: str, weight: int) -> None:
        self.name = name
        self.weight = weight

    def __str__(self) -> str:
        return f"{self.name} ({self.weight})"

    def __repr__(self) -> str:
        return f"State({self.name}, {self.weight})"

    def __eq__(self, other: 'State') -> bool:
        return self.name == other.name and self.weight == other.weight

    def __ne__(self, other: 'State') -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.name, self.weight))


if __name__ == "__main__":
    with open("data/6.txt", "r") as f:
        table = TransportationTable.from_file(f)
    print("Costs matrix :")
    table.show(table.costs)
    table.NorthWestCorner()
    print("Transportation table :")
    table.show(table.transportation_table)
    potentials = table.potentials()
    print("Marginal costs :")
    table.show(table.marginal_costs, potentials[0], potentials[1])
    min_mcost = min(table.marginal_costs)
    print(f"Minimum marginal cost is {min_mcost} at {table.marginal_costs.index(min_mcost)}")
    table.BalasHammer()
    print("Balas-Hammer algorithm :")
    table.show(table.transportation_table)
    potentials = table.potentials()
    print("Marginal costs :")
    table.show(table.marginal_costs, potentials[0], potentials[1])
    min_mcost = min(table.marginal_costs)
    print(f"Minimum marginal cost is {min_mcost} at {table.marginal_costs.index(min_mcost)}")
    graph = table.get_graph()
    print("Graph :")
    print(graph)
    graph.display()
    print(f"Is the graph degenerate ? {graph.is_degenerate()}")
    print(f"Does the graph have a cycle ? {graph.has_cycle()}")
    # wait for the user to continue
    input("Press any key to continue...")
    # add a cycle by adding an edge with the marginal cost
    index = table.marginal_costs.index(min_mcost)
    state = State(f"P_{index.row}", table.supply[index.row])
    next_state = State(f"O_{index.col}", table.demand[index.col])
    graph.edges.append((state, next_state, 0))
    graph.display()
    print(f"Is the graph degenerate ? {graph.is_degenerate()}")
    print(f"Does the graph have a cycle ? {graph.has_cycle()}")
