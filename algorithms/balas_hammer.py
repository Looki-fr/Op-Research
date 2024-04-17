
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

def balas_hammer_algorithm(transportation_table):
    num_sources = len(transportation_table) - 1
    num_destinations = len(transportation_table[0]) - 1
    
    supply = [int(transportation_table[i][-1]) for i in range(num_sources)]
    demand = [int(transportation_table[-1][j]) for j in range(num_destinations)]
    costs = [[int(transportation_table[i][j]) for j in range(num_destinations)] for i in range(num_sources)]
    
    allocations = [[0] * num_destinations for _ in range(num_sources)]
    
    while True:
        penalties = calculate_penalties(costs, allocations, supply, demand)
        if not penalties:
            break
        
        display_max_penalty(penalties)
        
        edge_to_fill = choose_edge_to_fill(penalties, costs, allocations, supply, demand)
        if edge_to_fill is None:
            break
        
        fill_edge(allocations, edge_to_fill, supply, demand)
    
    return allocations