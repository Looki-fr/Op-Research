import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

points = {
    10: 3.111,
    40: 57.309,
    100: 254.753,
    200: 1439.482,
    400: 10365.372,
    # 1000: 147521.924,
    1000: 153088.374,
    4000: 11331082.928
}


# generate log points for points
x = np.linspace(10, 400, 1000)
log = np.log(x)
exp = np.exp(x)
quad = x ** 2
# plt.plot(x, log, label="log(x)")
# plt.plot(x, exp, label="exp(x)")
# plt.plot(x, quad, label="x^2")

# Extracting x and y values from points dictionary
x = np.array(list(points.keys()))
y = np.array(list(points.values()))


def linear_fit(x, y):
    # Fitting a linear regression line
    coefficients = np.polyfit(x, y, 1)
    poly = np.poly1d(coefficients)

    # Generating points for the regression line
    x_reg = np.linspace(0, 450, 1000)
    y_reg = poly(x_reg)

    plt.plot(x_reg, y_reg, label="Linear (fitted)", color="red")


def quadratic_fit(x, y):
    # Fitting a quadratic regression line
    coefficients = np.polyfit(x, y, 2)
    print(coefficients)
    poly = np.poly1d(coefficients)

    # Generating points for the regression line
    x_reg = np.linspace(x[0], x[-1], 1000)
    y_reg = poly(x_reg)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="Quadratic (fitted)", color="green")


def exponential_fit(x, y):
    # Define the exponential function
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    # Fitting the exponential function
    popt, pcov = curve_fit(exponential_func, x, y)

    # Generating points for the regression line
    x_reg = np.linspace(x[0], x[-1], 1000)
    y_reg = exponential_func(x_reg, *popt)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="Exponential Regression")


def log_fit(x, y):
    # Define the logarithmic function
    def log_func(x, a, b):
        return a * np.log(b * x)

    # Fitting the logarithmic function
    popt, pcov = curve_fit(log_func, x, y)

    # Generating points for the regression line
    x_reg = np.linspace(x[0], x[-1], 1000)
    y_reg = log_func(x_reg, *popt)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="Logarithmic Regression")


def poly_fit(x, y):
    # Fitting a polynomial regression line
    coefficients = np.polyfit(x, y, 3)
    print(coefficients)
    poly = np.poly1d(coefficients)

    print(poly(4000))

    # Generating points for the regression line
    x_reg = np.linspace(x[0], x[-1], 1000)
    y_reg = poly(x_reg)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="Polynomial (fitted)", color="purple")


quadratic_fit(x, y)
poly_fit(x, y)
plt.scatter(*zip(*points.items()), label="Balas-Hammer")

plt.xlabel("Problem size")
plt.ylabel("Time (ms)")
plt.legend()
plt.title("Balas-Hammer Algorithm Performance")
plt.tight_layout()

plt.show()
