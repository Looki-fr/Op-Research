from collections import deque
class Problem:
    def __init__(self, transportation_table):
        self.transportation_table = transportation_table
        self.num_sources = len(transportation_table) - 1
        self.num_destinations = len(transportation_table[0]) - 1
        
        self.supply = [int(transportation_table[i][-1]) for i in range(self.num_sources)]
        self.demand = [int(transportation_table[-1][j]) for j in range(self.num_destinations)]
        self.costs = [[int(transportation_table[i][j]) for j in range(self.num_destinations)] for i in range(self.num_sources)]
        self.allocations = [[0] * self.num_destinations for _ in range(self.num_sources)]

    def get_total_cost(self):
        total_cost = 0
        for i in range(self.num_sources):
            for j in range(self.num_destinations):
                total_cost += self.costs[i][j] * self.allocations[i][j]
        
        return total_cost

    def get_graph(self):
        graph = {}
        for i in range(self.num_sources):
            for j in range(self.num_destinations):
                if self.allocations[i][j] > 0:
                    if i == 0 and j == 0:
                        print("source",i+1,"dest",j+1)
                    graph[f"S{i+1}"]=graph.get(f"S{i+1}",[])+[(f"C{j+1}",self.allocations[i][j])]
                    graph[f"C{j+1}"]=graph.get(f"C{j+1}",[])+[(f"S{i+1}",self.allocations[i][j])]

        return graph
    
    def is_acyclic(self):
        graph = self.get_graph()
        visited = set()
        parent = {node: None for node in graph}  # Initialize all nodes with None

        for node in graph:
            if node not in visited:
                queue = deque([node])

                while queue:
                    current_node = queue.popleft()
                    if current_node==None:
                        continue
                    if current_node in visited:
                        if parent[current_node] is not None and parent[current_node] != current_node:
                            cycle = [current_node]
                            while current_node!=None and parent[current_node] != current_node:
                                cycle.append(parent[current_node])
                                current_node = parent[current_node]
                            cycle.reverse()
                            return True, cycle[1:]
                    else:
                        visited.add(current_node)
                        for neighbor, _ in graph[current_node]:
                            if neighbor not in visited:
                                parent[neighbor] = current_node
                                queue.append(neighbor)

        return False, None
    