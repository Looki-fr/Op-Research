import matplotlib.pyplot as plt
import numpy as np

bh_init_points = {
    10: 3.111,
    40: 57.309,
    100: 254.753,
    400: 10365.372
}

bh_opti_points = {
    10: 27.022,
    40: 1284.384,
    100: 9061.155,
    400: 254663.324
}

nw_init_points = {
    10: 0.057,
    40: 0.19,
    100: 0.702,
    400: 2.745
}

nw_opti_points = {
    10: 150.902,
    40: 5288.096,
    100: 56244.751,
    400: 1297231.005
}

# y = (nw init + nw opti) / (bh init + bh opti)
nw_point = [
    [k, v1 + v2] for k, v1, v2 in zip(nw_init_points.keys(), nw_init_points.values(), nw_opti_points.values())
]

bh_point = [
    [k, v1 + v2] for k, v1, v2 in zip(bh_init_points.keys(), bh_init_points.values(), bh_opti_points.values())
]

index = np.array([k for k, _ in nw_point])
nw_y = np.array([v for _, v in nw_point])
bh_y = np.array([v for _, v in bh_point])

y = nw_y / bh_y

print(index)
print(nw_y)
print(bh_y)

plt.scatter(index, y, label="Factor")


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
plt.xticks(index)
plt.ylim(2, 8)
plt.xlabel("Problem size")
plt.ylabel("Factor of Improvement")
plt.legend()
plt.title("Factor of Improvement of Balas-Hammer Algorithm\nover North-West Corner Algorithm")
plt.tight_layout()

plt.show()

# make a scatter for each worst time for each size with both algorithms
plt.scatter(index, [v for _, v in nw_point], label="North-West Corner")
plt.scatter(index, [v for _, v in bh_point], label="Balas-Hammer")
plt.xlabel("Problem size")
plt.ylabel("Time (ms)")
plt.legend()

plt.title("Worst Times for North-West Corner and Balas-Hammer Algorithms")
plt.tight_layout()

plt.show()
