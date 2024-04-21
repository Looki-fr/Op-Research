from utils_functions import *
from algorithms.balas_hammer import balas_hammer_algorithm
from algorithms.north_west import north_west_corner_method
from algorithms.stepping_stone_enhanced import stepping_stone_method
from algorithms.total_cost import get_total_cost

def main():
    # table de l'exercice 2 du TD
    transportation_table = [
        [90,70,110,60,75],
        [120,60,50,70,40],
        [80,100,70,80,70],
        [65,40,50,30,185]
    ]
    print_transportation_table(transportation_table)
    allocations = balas_hammer_algorithm(transportation_table)
    print_table(allocations)
    print("Total cost: ", get_total_cost(transportation_table, allocations))
    
    # allocations = north_west_corner_method(transportation_table)
    # print_table(allocations)
    
    # costs = [[19, 30, 50, 10], [70, 30, 40, 60], [40, 8, 70, 20], [60, 40, 80, 30]]
    # supply = [300, 400, 200, 500]
    # demand = [200, 300, 400, 300]
    # allocations = stepping_stone_method(costs, supply, demand)
    # print_allocation_table(allocations)
    

if __name__ == "__main__":
    main()