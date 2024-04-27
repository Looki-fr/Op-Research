from tools import *
from random import randint
from timer import Timer


def generate_problem(n, m):
    problem = TransportationTable()
    matrix = Matrix(n, m)
    for i in range(n):
        for j in range(m):
            matrix[Index(i, j)] = randint(1, 100)
    problem.costs = matrix
    temp = Matrix(n, m)
    for i in range(n):
        for j in range(m):
            temp[Index(i, j)] = randint(1, 100)
    problem.supply = [sum(temp.rows[i]) for i in range(n)]
    problem.demand = [sum(temp.cols[j]) for j in range(m)]
    if sum(problem.supply) != sum(problem.demand):
        raise ValueError("The sum of supply and demand must be equal")
    return problem


if __name__ == "__main__":
    for i in range(100):
        print(f"Test {i + 1}")
        n = 40
        problem = generate_problem(n, n)
        #!print(problem)
        problem.NordWestOptimized()
        problem.BalasHammerOptimized()
    print("Average times:\n - ", end="")
    print(*[f"{k}: {round(v * 1000, 3)} ms (" + str(len(Timer.timedict[k])) + ")" for k, v in Timer.average_times.items()], sep="\n - ")
