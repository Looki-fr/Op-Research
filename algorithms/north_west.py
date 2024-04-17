def north_west_corner_method(transportation_table):
    num_sources = len(transportation_table) - 1
    num_destinations = len(transportation_table[0]) - 1
    
    supply = [int(transportation_table[i][-1]) for i in range(num_sources)]
    demand = [int(transportation_table[-1][j]) for j in range(num_destinations)]
    
    allocations = [[0] * num_destinations for _ in range(num_sources)]
    
    supply_remaining = supply[:]
    demand_remaining = demand[:]
    
    i = 0
    j = 0
    
    while i < num_sources and j < num_destinations:
        if supply_remaining[i] == 0:
            i += 1
            continue
        if demand_remaining[j] == 0:
            j += 1
            continue
        
        allocation = min(supply_remaining[i], demand_remaining[j])
        allocations[i][j] = allocation
        
        supply_remaining[i] -= allocation
        demand_remaining[j] -= allocation
        
        if supply_remaining[i] == 0:
            i += 1
        if demand_remaining[j] == 0:
            j += 1
    
    return allocations