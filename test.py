from tools import *
from random import randint
from timer import Timer


def generate_problem(n, m):
    problem = TransportationTable()
    matrix = Matrix(n, m)
    for i in range(n):
        for j in range(m):
            matrix[(i, j)] = randint(1, 100)
    problem.costs = matrix
    temp = Matrix(n, m)
    for i in range(n):
        for j in range(m):
            temp[(i, j)] = randint(1, 100)
    problem.supply = [sum(temp.rows[i]) for i in range(n)]
    problem.demand = [sum(temp.cols[j]) for j in range(m)]
    if sum(problem.supply) != sum(problem.demand):
        raise ValueError("The sum of supply and demand must be equal")
    return problem


if __name__ == "__main__":
    for i in range(100):
        print(f"Test {i + 1}")
        n = 10
        problem = generate_problem(n, n)
        print("Problem generated")
        #!print(problem)
        problem.BalasHammerOptimized()
        print("BalasHammerOptimized")
        problem.NordWestOptimized()
        print("NordWestOptimized")
    print("Average times:\n - ", end="")
    print(*[f"{k}: {round(v * 1000, 3)} ms (" + str(len(Timer.timedict[k])) + ")" for k, v in Timer.average_times.items()], sep="\n - ")
    print("Worst times:\n - ", end="")
    print(*[f"{k}: {round(v * 1000, 3)} ms" for k, v in Timer.worst_times.items()], sep="\n - ")
