from utils_functions import *
from algorithms.balas_hammer import balas_hammer_algorithm
from algorithms.north_west import north_west_corner_method
from algorithms.stepping_stone_final import stepping_stone
from problem import Problem

def main():
    # table de l'exercice 2 du TD
    transportation_table = [
        [90,70,110,60,75],
        [120,60,50,70,40],
        [80,100,70,80,70],
        [65,40,50,30,185]
    ]
    # table de l'exercice 1 du TD (avec cycle)
    transportation_table = [
        [11,12,10,10,60],
        [17,16,15,18,30],
        [19,21,20,22,90],
        [50,75,30,25,180]
    ]
    problem = Problem(transportation_table)
    print_transportation_table(problem.transportation_table)
    print(problem.costs)
    balas_hammer_algorithm(problem)
    print(problem.costs)
    print_table(problem.allocations)
    print("Total cost: ", problem.get_total_cost())
    print(problem.get_graph())
    print("Is acyclic: ", problem.is_acyclic())

if __name__ == "__main__":
    main()