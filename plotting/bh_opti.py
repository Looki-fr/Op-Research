import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

points = {
    10: 27.022,
    40: 1284.384,
    100: 9061.155,
    400: 254663.324
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
    poly = np.poly1d(coefficients)

    # Generating points for the regression line
    x_reg = np.linspace(10, 400, 1000)
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
    x_reg = np.linspace(10, 400, 1000)
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
    x_reg = np.linspace(10, 400, 1000)
    y_reg = log_func(x_reg, *popt)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="Logarithmic Regression")


def nlogn_fit(x, y):
    # Define the nlogn function
    def nlogn_func(x, a, b):
        return a * x * np.log(b * x)

    # Fitting the nlogn function
    popt, pcov = curve_fit(nlogn_func, x, y)

    # Generating points for the regression line
    x_reg = np.linspace(10, 400, 1000)
    y_reg = nlogn_func(x_reg, *popt)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="nlogn Regression")


def poly_fit(x, y):
    # Fitting a polynomial regression line
    coefficients = np.polyfit(x, y, 3)
    poly = np.poly1d(coefficients)

    # Generating points for the regression line
    x_reg = np.linspace(10, 400, 1000)
    y_reg = poly(x_reg)

    # Plotting the regression line
    plt.plot(x_reg, y_reg, label="Polynomial (fitted)", color="purple")


exponential_fit(x, y)
plt.scatter(*zip(*points.items()), label="North-West Corner")

plt.xlabel("Problem size")
plt.ylabel("Time (ms)")
plt.legend()
plt.title("North-West Corner Algorithm Performance")
plt.tight_layout()

plt.show()
