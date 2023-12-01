import numpy as np


def problem_1(x1, x2):
    return ((x1 ** 2) + (x2 ** 2))


def problem_2(x1, x2):
    return np.exp(-((x1 ** 2) + (x2 ** 2))) + 2 * np.exp(-(((x1 - 1.7) ** 2) + ((x2 - 1.7) ** 2)))


def problem_3(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * ((x1 ** 2) + (x2 ** 2)))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)


def problem_4(x1, x2):
    return ((x1 ** 2) - 10 * np.cos(2 * np.pi * x1) + 10) + ((x2 ** 2) - 10 * np.cos(2 * np.pi * x2) + 10)


def problem_5(x1, x2):
    return ((x1 - 1) ** 2) + 100 * ((x2 - (x1 ** 2)) ** 2)


def problem_6(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin((4 * np.pi * x2) + np.pi) + 1


def problem_7(x1, x2):
    return -np.sin(x1) * (np.sin((x1 ** 2) / np.pi) ** (2 * 10)) -np.sin(x2) * (np.sin((2 * (x2 ** 2)) / np.pi) ** (2 * 10))


def problem_8(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

