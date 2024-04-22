
def calculate_penalties(costs, supply, demand):
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

def choose_edge_to_fill(penalties, costs, allocations, supply, demand):
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
                print(f"{penalty_type}: {cell}, Penalty: {penalty}")
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
                print(f"{penalty_type}: {cell}, Penalty: {penalty}")
                return (min_cost_index, i)
    
    return None

def fill_edge(allocations, edge, supply, demand, costs):
    i, j = edge
    allocation = min(supply[i], demand[j])
    allocations[i][j] = allocation
    supply[i] -= allocation
    demand[j] -= allocation

    if supply[i] == 0:
        for k in range(len(costs[i])):
            costs[i][k] = float('inf')
    if demand[j] == 0:
        for k in range(len(costs)):
            costs[k][j] = float('inf')

def find_last_edge_to_fill(allocations, supply, demand):
    for i in range(len(allocations)):
        for j in range(len(allocations[0])):
            if allocations[i][j] == 0 and supply[i] > 0 and demand[j] > 0:
                return (i, j)
    return None

def balas_hammer_algorithm(problem):
    temp_costs=[row[:] for row in problem.costs]
   
    while True:
        edge_to_fill=None
        penalties = calculate_penalties(temp_costs, problem.supply, problem.demand)
        if not penalties:
            edge_to_fill = find_last_edge_to_fill(problem.allocations, problem.supply, problem.demand)
            if edge_to_fill is None:
                print("no more penalties")
                print("no more edges to fill")
                break
        
        
        if edge_to_fill == None:
            edge_to_fill = choose_edge_to_fill(penalties, temp_costs, problem.allocations, problem.supply, problem.demand)

        if edge_to_fill is None:
            print("no more edges to fill")
            break
        
        fill_edge(problem.allocations, edge_to_fill, problem.supply, problem.demand, temp_costs)
