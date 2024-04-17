from collections import deque

def is_acyclic(graph):
    for vertex in graph:
        if not bfs_cycle_detection(graph, vertex):
            return False
    return True

def bfs_cycle_detection(graph, start_vertex):
    visited = set()
    queue = deque([(start_vertex, None)])
    
    while queue:
        vertex, parent = queue.popleft()
        visited.add(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append((neighbor, vertex))
            elif neighbor != parent:
                return False
    
    return True

def find_cycle(graph):
    visited = set()
    stack = []
    
    def dfs(vertex):
        if vertex in stack:
            return stack[stack.index(vertex):]
        
        if vertex in visited:
            return None
        
        visited.add(vertex)
        stack.append(vertex)
        
        for neighbor in graph.get(vertex, []):
            cycle = dfs(neighbor)
            if cycle is not None:
                return cycle
        
        stack.pop()
        return None
    
    for vertex in graph:
        cycle = dfs(vertex)
        if cycle is not None:
            return cycle
    
    return None


def calculate_penalties(costs, allocations, supply, demand):
    penalties = []
    for i in range(len(costs)):
        if sum(allocations[i]) < supply[i]:
            row_costs = costs[i][:]
            row_costs.sort()
            penalty = row_costs[1] - row_costs[0]
            penalties.append(((i), penalty, "row"))
        
        column_allocations = [row[i] for row in allocations]
        if sum(column_allocations) < demand[i]:
            column_costs = [costs[x][i] for x in range(len(costs))]
            column_costs.sort()
            penalty = column_costs[1] - column_costs[0]
            penalties.append(((i), penalty, "column"))
    
    penalties.sort(key=lambda x: x[1], reverse=True)
    return penalties

def display_max_penalty(penalties):
    max_penalty = penalties[0][1]
    max_penalties = [item for item in penalties if item[1] == max_penalty]
    for cell, penalty, type in max_penalties:
        print(f"{type}: {cell}, Penalty: {penalty}")

def choose_edge_to_fill(penalties, costs, allocations, supply, demand):
    max_penalty = penalties[0][1]
    
    for cell, penalty, penalty_type in penalties:
        i = cell
        if penalty_type == "row":
            min_cost = float('inf')
            min_cost_index = None
            for j, cost in enumerate(costs[i]):
                if supply[i] > 0 and demand[j] > 0 and allocations[i][j] == 0:
                    if cost < min_cost:
                        min_cost = cost
                        min_cost_index = j
            if min_cost_index is not None:
                return (i, min_cost_index)
        elif penalty_type == "column":
            min_cost = float('inf')
            min_cost_index = None
            for k, row in enumerate(costs):
                if supply[k] > 0 and demand[i] > 0 and allocations[k][i] == 0:
                    if row[i] < min_cost:
                        min_cost = row[i]
                        min_cost_index = k
            if min_cost_index is not None:
                return (min_cost_index, i)
    
    return None

def fill_edge(allocations, edge, supply, demand):
    i, j = edge
    allocation = min(supply[i], demand[j])
    allocations[i][j] = allocation
    supply[i] -= allocation
    demand[j] -= allocation

def compute_potentials(costs, allocations):
    n, m = len(costs), len(costs[0])
    u = [None] * n
    v = [None] * m
    visited = [[False] * m for _ in range(n)]
    visited[0][0] = True
    
    u[0] = 0
    while None in u or None in v:
        for i in range(n):
            for j in range(m):
                if visited[i][j] and allocations[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = costs[i][j] - u[i]
                    elif u[i] is None and v[j] is not None:
                        u[i] = costs[i][j] - v[j]
                    for i2 in range(n):
                        for j2 in range(m):
                            if not visited[i2][j2] and (i2 == i or j2 == j):
                                visited[i2][j2] = True
    return u, v

def stepping_stone_method(costs, supply, demand):
    n, m = len(costs), len(costs[0])
    allocations = [[0] * m for _ in range(n)]
    
    while True:
        u, v = compute_potentials(costs, allocations)
        print("u:", u)
        reduced_costs = [[costs[i][j] - u[i] - v[j] for j in range(m)] for i in range(n)]
        
        graph = {}
        for i in range(n):
            for j in range(m):
                if allocations[i][j] == 0:
                    graph[(i, j)] = []
                    if i > 0 and allocations[i - 1][j] > 0:
                        graph[(i, j)].append((i - 1, j))
                    if i < n - 1 and allocations[i + 1][j] > 0:
                        graph[(i, j)].append((i + 1, j))
                    if j > 0 and allocations[i][j - 1] > 0:
                        graph[(i, j)].append((i, j - 1))
                    if j < m - 1 and allocations[i][j + 1] > 0:
                        graph[(i, j)].append((i, j + 1))
        
        if is_acyclic(graph):
            break
        
        cycle = find_cycle(graph)
        min_allocation = min(allocations[i][j] for i, j in cycle[1::2])
        
        for i, j in cycle[1::2]:
            allocations[i][j] -= min_allocation
        for i, j in cycle[::2]:
            allocations[i][j] += min_allocation
    
    return allocations