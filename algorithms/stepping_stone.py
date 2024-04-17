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

def stepping_stone_method(costs, supply, demand):
    graph = {}
    for i in range(len(costs)):
        for j in range(len(costs[0])):
            graph[(i, j)] = []
    
    for i in range(len(costs)):
        for j in range(len(costs[0])):
            if supply[i] > 0 and demand[j] > 0:
                graph[(i, j)].append((i, len(costs[0]) + j))
                if (i, len(costs[0]) + j) not in graph:
                    graph[(i, len(costs[0]) + j)] = []
                graph[(i, len(costs[0]) + j)].append((i, j))
    
    if not is_acyclic(graph):
        return None
    
    allocations = [[0 for _ in range(len(costs[0]))] for _ in range(len(costs))]
    while True:
        penalties = calculate_penalties(costs, allocations, supply, demand)
        if not penalties:
            break
        
        display_max_penalty(penalties)
        edge = choose_edge_to_fill(penalties, costs, allocations, supply, demand)
        if edge is None:
            break
        
        fill_edge(allocations, edge, supply, demand)
    
    return allocations

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

    
