import matplotlib.pyplot as plt
import numpy as np

points = {
    10: 0.057,
    40: 0.19,
    100: 0.702,
    400: 2.745
}


# generate log points for points
x = np.linspace(10, 400, 1000)
log = np.log(x)
exp = np.exp(x)
quad = x ** 2
# plt.plot(x, log, label="log(x)")
# plt.plot(x, exp, label="exp(x)")
# plt.plot(x, quad, label="x^2")

# make line for Balas-Hammer points
x = np.linspace(10, 400, 1000)

# Extracting x and y values from points dictionary
x = np.array(list(points.keys()))
y = np.array(list(points.values()))

# Fitting a linear regression line
coefficients = np.polyfit(x, y, 1)
poly = np.poly1d(coefficients)

# Generating points for the regression line
x_reg = np.linspace(0, 450, 1000)
y_reg = poly(x_reg)

plt.plot(x_reg, y_reg, label="Linear (fitted)", color="red")
plt.scatter(*zip(*points.items()), label="North-West Corner")

plt.xlabel("Problem size")
plt.ylabel("Time (ms)")
plt.legend()
plt.title("North-West Corner Algorithm Performance")
plt.tight_layout()

plt.show()
