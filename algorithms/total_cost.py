def get_total_cost(transportation_table, allocations):
    num_sources = len(transportation_table) - 1
    num_destinations = len(transportation_table[0]) - 1
    costs = [[int(transportation_table[i][j]) for j in range(num_destinations)] for i in range(num_sources)]
    total_cost = 0
    for i in range(len(costs)):
        for j in range(len(costs[i])):
            total_cost += costs[i][j] * allocations[i][j]
    return total_cost