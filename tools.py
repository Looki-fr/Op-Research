from io import TextIOWrapper
from matplotlib.image import resample
from tabulate import tabulate
from typing import Union
from logger import print
from numpy import linalg
import graphviz as gv
import string
from random import Random, shuffle
import time
from timer import Timer
import networkx as nx
from networkx.classes import Graph as nxGraph
from networkx.algorithms import cycles
from networkx.exception import NetworkXNoCycle


# define for typing
class Edge(tuple):
    def __init__(self) -> tuple[int, int]:
        super().__init__()


def char_map(x): return char_map(x // 26) + string.ascii_lowercase[x % 26] if x // 26 else string.ascii_lowercase[x]


def translate(x: tuple[int, int]) -> tuple[str, str]: return (f"S_{x[0] + 1}", f"C_{x[1] + 1}")


def untranslate(x: tuple[str, str]) -> tuple[int, int]:
    # remove elements after the 2 first elements of the tuple
    x = x[:2]
    x = sorted(x, key=lambda x: x[0], reverse=True)
    return (int(x[0][2:]) - 1, int(x[1][2:]) - 1)


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

    def __setitem__(self, index: tuple[int, int], value: int):
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
    _graph: 'Graph' = None

    @property
    def random(self) -> Random:
        return Random(self.seed)

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

    @Timer.timeit_with_name("0_nw")
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

    def _get_indexes(self) -> list[tuple[int, int]]:
        # list the indexes where the values are not null in the transportation table
        indexes: list[tuple[int, int]] = []
        for i, row in enumerate(self.transportation_table.rows):
            for j, cell in enumerate(row):
                if cell != 0:
                    indexes.append((i, j))
        return indexes

    def _get_missing_indexes(self, indexes: list[tuple[int, int]]) -> None:
        missing_indexes = []
        indexes: set = set(indexes)  # Convert to set for faster membership check
        supply_length = len(self.supply)
        demand_length = len(self.demand)
        nb_missing_indexes = supply_length + demand_length - len(indexes) - 1

        if self.missing_edge_buffer is not None and nb_missing_indexes > 0:
            missing_indexes.append(self.missing_edge_buffer)
            indexes.add(self.missing_edge_buffer)

        # # Initialize random generator with the class seed
        gn = self.random
        all_indexes = {(i, j) for i in range(supply_length) for j in range(demand_length)}

        # Shuffle the edges randomly
        shuffled_indexes = list(all_indexes - indexes)
        gn.shuffle(shuffled_indexes)

        # Sort the shuffled edges by cost
        sorted_indexes = sorted(shuffled_indexes, key=lambda x: self.costs[x])

        # add the missing edges without creating a cycle
        graph = Graph(self.graph)
        # add the indexes we already have
        graph.add_edges_from(map(translate, indexes))
        for index in sorted_indexes:
            if len(missing_indexes) == nb_missing_indexes:
                break
            edge = translate(index)
            graph.add_edge(*edge)
            if not graph.has_cycle():
                missing_indexes.append(index)
            else:
                graph.remove_edge(*edge)

        if len(missing_indexes) != nb_missing_indexes:
            raise ValueError("Error in the missing edges allocation")

        return missing_indexes

    def potentials(self) -> tuple[list[int], list[int]]:
        if self.graph.is_degenerate():
            self.graph.display()
            raise ValueError("The graph is degenerate")
        size = len(self.supply) + len(self.demand)
        # potentials are binds by the following equation
        #  c_ij = s_i - t_j
        s = [None] * len(self.supply)
        t = [None] * len(self.demand)
        # make the system of equations matrix to solve
        matrix = Matrix(size, size)
        costs = []
        edges = self.graph.edges
        indexes = map(untranslate, edges)  # TODO : this maybe isn't very efficient to do this at each iteration
        # fill the matrix
        for i, index in enumerate(indexes):
            matrix[(i, index[0])] = 1
            matrix[(i, len(self.supply) + index[1])] = -1
            costs.append(self.costs[index])
        else:
            # init one of the potentials to 0
            matrix[(size - 1, 0)] = 1
            costs.append(0)
        try:
            res = linalg.solve(matrix.matrix, costs)
        except linalg.LinAlgError:
            self.graph.display()
            raise ValueError("The system of equations is not solvable")
        # assign the values to the potentials
        s = [*map(int, res[:len(self.supply)])]
        t = [*map(int, res[len(self.supply):])]
        return (s, t)

    @Timer.timeit_with_name("0_bh")
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

    def _init_graph(self) -> 'Graph':
        graph = Graph()
        indexes = self._get_indexes()
        # translate tuple to edges
        edges = map(translate, indexes)
        # add the nodes
        graph.add_edges_from(edges)
        return graph

    @property
    def graph(self) -> 'Graph':
        if self._graph:
            return self._graph
        self._graph = self._init_graph()
        return self._graph

    def optimize(self) -> bool:
        # check if there is a cycle in the actual proposition
        first = True

        while True:
            # Step 1 : Randomize the seed to get a different result each iteration and avoid getting stuck in a loop
            self.seed = time.time()

            # Step 2 : Check if the graph has a cycle and remove it
            # ? isn't mandatory to do at each iteration
            if self.graph.is_degenerate():
                if cycle := self.graph.has_cycle():
                    cycle = list(map(untranslate, cycle))
                    self.stepping_stone(cycle)
                else:
                    # add the missing edges until the graph is connected
                    edges = self._get_missing_indexes(self._get_indexes())
                    self.graph.add_edges_from(map(translate, edges))
            # Step 3 : Compute the marginal costs
            marginal_costs = self.marginal_costs
            min_mcost = min(marginal_costs)
            # Step 4 : Check if the table is optimized or not with the minimum marginal cost
            if min_mcost >= 0:
                self._graph = None
                return first
            # Step 5 : Add the edge with the minimum marginal cost to the graph and use the stepping stone method
            index = marginal_costs.index(min_mcost)
            # add the edge to the graph
            edge = translate(index)
            self.graph.add_edge(*edge)
            # find the added cycle
            cycle: list[Edge] = self.graph.get_cycle(edge)
            cycle = list(map(untranslate, cycle))
            #!print("Cycle : ", end='')
            #!print(*cycle, sep=' -> ')
            # optimize the table
            delta = self.stepping_stone(cycle)
            if delta == 0:
                self.missing_edge_buffer = index
            else:
                self.missing_edge_buffer = None
            #!print(f"Delta : {delta}")
            #!self.show(self.transportation_table)
            first = False

    @property
    def total_cost(self) -> int:
        return sum([self.costs[index] * self.transportation_table[index] for index in self._get_indexes()])

    def stepping_stone(self, cycle: list['Edge']) -> int:
        delta_cost = [self.costs[edge] * (-1)**i for i, edge in enumerate(cycle)]
        delta_cost = sum(delta_cost)
        if delta_cost > 0:
            delta = -min([self.transportation_table[edge] for edge in cycle if cycle.index(edge) % 2 == 0])
        elif delta_cost < 0:
            delta = min([self.transportation_table[edge] for edge in cycle if cycle.index(edge) % 2 == 1])
        else:
            raise ValueError("The cost didn't change")
        if delta == 0:
            for i, edge in enumerate(cycle):
                if self.transportation_table[edge] == 0:
                    self.graph.remove_edge(*translate(edge))
            return 0
        for i, edge in enumerate(cycle):
            if i % 2 == 0:
                self.transportation_table[edge] += delta
            else:
                self.transportation_table[edge] -= delta
            # ? remove the edge if the value is 0
            if self.transportation_table[edge] == 0:
                self.graph.remove_edge(*translate(edge))
        return delta

    def NordWestOptimized(self):
        self.NorthWestCorner()
        t1 = Timer.get_time()
        self.optimize()
        Timer.time(t1, "t_nw")

    def BalasHammerOptimized(self):
        self.BalasHammer()
        t1 = Timer.get_time()
        self.optimize()
        Timer.time(t1, "t_bh")


class Graph(nxGraph):
    def __repr__(self) -> str:
        return f"Graph {id(self)}"

    def display(self, cycle: list['Edge'] = []) -> None:
        """
        Display the graph using graphviz
        """
        graph = gv.Digraph()
        for node in self.nodes():
            graph.node(str(node))
        for edge in self.edges():
            color = "black"
            state, next_state = edge
            if edge in cycle or edge[::-1] in cycle:
                color = "red"
            graph.edge(str(state), str(next_state), arrowhead="none", color=color)
        graph.view(cleanup=True)
        input("Press a key...")

    def is_degenerate(self) -> bool:
        return not nx.is_connected(self) or len(self.edges) != len(self.nodes) - 1

    def has_cycle(self, start=None) -> Union[list[tuple[int, int]], bool]:
        try:
            return cycles.find_cycle(self, source=start, orientation='ignore')
        except NetworkXNoCycle:
            return False

    def get_cycle(self, start=None) -> Union[list[tuple[int, int]], bool]:
        return cycles.find_cycle(self, source=start, orientation='ignore')


if __name__ == "__main__":

    costs_nordwest = {}
    costs_balas = {}
    costs_test = {}

    def test_transportation_table(i):
        print(f"Test {i}")
        with open(f"data/{i}.txt", "r") as f:
            table = TransportationTable.from_file(f)
        table.BalasHammer()
        table.show(table.transportation_table)
        costs_nordwest[i] = table.total_cost

    test_transportation_table(9)

    # assert costs_nordwest == {1: 3000, 2: 2000, 3: 33000, 4: 12700, 5: 445, 6: 2880, 7: 16000, 8: 17600, 9: 5700, 10: 54000, 11: 279150, 12: 154400}
    # print("Nordwest algorithm is working")
    # assert costs_balas == {1: 3000, 2: 2000, 3: 33000, 4: 12700, 5: 445, 6: 2880, 7: 16000, 8: 17600, 9: 5700, 10: 54000, 11: 279150, 12: 154400}
    # print("Balas algorithm is working")
