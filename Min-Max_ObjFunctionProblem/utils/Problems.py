import numpy as np


def first_problem(x_1, x_2):
    """Find the minimun function value for this problem."""
    return ((x_1 ** 2) + (x_2 ** 2))


def second_problem(x_1, x_2):
    """Find the maximum function value for this problem."""
    return ((np.e ** -(x_1 ** 2 + x_2 ** 2)) + 2 * (np.e ** -(((x_1 - 1.7) ** 2 + (x_2 - 1.7) ** 2))))


def third_problem(x_1, x_2):
    """Find the minimun function value for this problem."""
    return (-20 * (np.e ** (-0.2 * (np.sqrt(0.5 * (x_1 ** 2 + x_2 ** 2))))) - (np.e ** (0.5 * ((np.cos(2 * np.pi * x_1)) + (np.cos(2 * np.pi * x_2))))) + 20 + (np.e ** 1))


def fourth_problem(x_1, x_2):
    """Find the minimun function value for this problem."""
    return (((x_1 ** 2) - 10 * np.cos(2 * np.pi * x_1) + 10) + ((x_2 ** 2) - 10 * np.cos(2 * np.pi * x_2) + 10))


def fifth_problem(x_1, x_2):
    """Find the minimun function value for this problem."""
    return ((x_1 - 1) ** 2) + 100 * ((x_2 - (x_1 ** 2)) ** 2)


def sixth_problem(x_1, x_2):
    """Find the maximum function value for this problem."""
    return (x_1 * np.sin(4 * np.pi * x_1) - x_2 * np.sin(4 * np.pi * x_2 + np.pi) + 1)


def seventh_problem(x_1, x_2):
    """Find the minimun function value for this problem."""
    return -np.sin(x_1) * (np.sin((x_1 ** 2) / np.pi) ** (2 * 10)) -np.sin(x_2) * (np.sin((2 * (x_2 ** 2)) / np.pi) ** (2 * 10))


def eighth_problem(x_1, x_2):
    """Find the minimun function value for this problem."""
    return -(x_2 + 47) * (np.sin(np.sqrt(np.abs((x_1 / 2) + (x_2 + 47))))) - x_1 * (np.sin(np.sqrt(np.abs(x_1 - (x_2 + 47)))))

