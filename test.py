from tools import *
from random import randint
from timer import Timer
import matplotlib.pyplot as plt
import numpy as np


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
    def get_computation_times(n):
        print(f"Computing times for n={n}")
        for _ in range(100):
            problem = generate_problem(n, n)
            problem.NordWestOptimized()
            problem.BalasHammerOptimized()

    times: dict[dict[str:list]] = {}
    index = [10, 40, 100]

    for n in index:
        get_computation_times(n)
        times[n] = Timer.timedict.copy()
        # pass each time in s to ms
        for method in times[n].keys():
            times[n][method] = [time * 1000 for time in times[n][method]]
        print("Average times:\n - ", end="")
        print(*[f"{k}: {round(v * 1000, 3)} ms (" + str(len(Timer.timedict[k])) + ")" for k, v in Timer.average_times.items()], sep="\n - ")
        print("Worst times:\n - ", end="")
        print(*[f"{k}: {round(v * 1000, 3)} ms" for k, v in Timer.worst_times.items()], sep="\n - ")

        Timer.timedict = {}

    methods = times[10].keys()
    for method in methods:
        # make a scatter plot for each method of the computation times
        points = list()
        for size in index:
            for point in times[size][method]:
                points.append((size, point))
        plt.scatter(*zip(*points), label=method)
        # add a red point for the average for each size
        print(points)
        for size in index:
            plt.scatter(size, sum(times[size][method]) / len(times[size][method]), color="red")

    # points = ...
    # plt.scatter(*zip(*points), label="Balas-Hammer")
    # # add a red point for the average for each size
    # for size in index:
    #     _ = [point[1] for point in points if point[0] == size]
    #     plt.scatter(size, sum(_) / len(_), color="red")
    plt.legend()

    # Padding values
    padding = 5

    # Define a custom transformation function to map original ticks to evenly spaced values
    def transform_function(x):
        index_with_padding = np.concatenate(
            [[index[0] - padding], index, [index[-1] + padding]])
        transformed_values = np.linspace(0, len(index_with_padding) - 1, len(index_with_padding))
        return np.interp(x, index_with_padding, transformed_values)

    # Inverse transformation function (not used in this example)
    def inverse_transform_function(x):
        index_with_padding = np.concatenate(
            [[index[0] - padding], index, [index[-1] + padding]])
        transformed_values = np.linspace(0, len(index_with_padding) - 1, len(index_with_padding))
        return np.interp(x, transformed_values, index_with_padding)

    plt.xscale("function", functions=(transform_function, inverse_transform_function))
    plt.yscale("log")

    # naming the x axis
    plt.xlabel('Problem size')
    # naming the y axis
    plt.ylabel('Time (ms) (Log scale)')

    plt.xticks(index)
    plt.show()
