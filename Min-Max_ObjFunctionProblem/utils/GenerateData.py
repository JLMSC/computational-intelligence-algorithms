import numpy as np
import matplotlib.pyplot as plt


def generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_2_bounds, N):
    """
    Generates random values (domains) within lower and upper bound
    using the given function.
    
    Parameters
    ---
    f: function, The function to be used.
    x_1_candidate: int, The x_1 candidate, if any.
    x_2_candidate: int, The x_2 candidate, if any.
    x_1_bounds: list[int | float], The lower and
        upper domain bounds for x_1.
    x_2_bounds: list[int | float], The lower and
        upper domain bounds for x_2.
    N: int, The amount of samples.

    Returns
    ---
    If both candidates are provided:
        returns x_1_candidate, x_2_candidate, f(x_1_candidate, x_2_candidate)
    If no candidates are provided:
        returns x_1, x_2, f(x_1, x_2)
    """
    if x_1_candidate is not None and x_2_candidate is not None:
        return x_1_candidate, x_2_candidate, f(x_1_candidate, x_2_candidate)

    x_1_lower_bounds, x_1_upper_bounds = x_1_bounds
    x_2_lower_bounds, x_2_upper_bounds = x_2_bounds

    # TODO: This will need to have unique names, to get candidates coords or so, I guess.
    x_1 = np.linspace(x_1_lower_bounds, x_1_upper_bounds, N)
    x_2 = np.linspace(x_2_lower_bounds, x_2_upper_bounds, N)

    x_1, x_2 = np.meshgrid(x_1, x_2)    

    return x_1, x_2, f(x_1, x_2)


def problem_one(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-100, 100], N = 1000):
    """Find the minimun function value for this problem."""
    def f(x_1, x_2):
        return ((x_1 ** 2) + (x_2 ** 2))

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_1_bounds, N)


def problem_two(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-2, 4], x_2_bounds = [-2, 5], N = 1000):
    """Find the maximum function value for this problem."""
    def f(x_1, x_2):
        return ((np.e ** -(x_1 ** 2 + x_2 ** 2)) + 2 * (np.e ** -(((x_1 - 1.7) ** 2 + (x_2 - 1.7) ** 2))))

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_2_bounds, N)


def problem_three(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-8, 8], N = 1000):
    """Find the minimun function value for this problem."""
    def f(x_1, x_2):
        return (-20 * (np.e ** (-0.2 * (np.sqrt(0.5 * (x_1 ** 2 + x_2 ** 2))))) - (np.e ** (0.5 * ((np.cos(2 * np.pi * x_1)) + (np.cos(2 * np.pi * x_2))))) + 20 + (np.e ** 1))

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_1_bounds, N)


def problem_four(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-5.12, 5.12], N = 1000):
    """Find the minimun function value for this problem."""
    def f(x_1, x_2):
        return (((x_1 ** 2) - 10 * np.cos(2 * np.pi * x_1) + 10) + ((x_2 ** 2) - 10 * np.cos(2 * np.pi * x_2) + 10))

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_1_bounds, N)


def problem_five(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-2, 2], x_2_bounds = [-1, 3], N = 1000):
    """Find the minimun function value for this problem."""
    def f(x_1, x_2):
        return ((x_1 - 1) ** 2) + 100 * ((x_2 - (x_1 ** 2)) ** 2)

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_2_bounds, N)


def problem_six(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-1, 3], N = 1000):
    """Find the maximum function value for this problem."""
    def f(x_1, x_2):
        return (x_1 * np.sin(4 * np.pi * x_1) - x_2 * np.sin(4 * np.pi * x_2 + np.pi) + 1)

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_1_bounds, N)


def problem_seven(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [0, np.pi], N = 1000):
    """Find the minimun function value for this problem."""
    def f(x_1, x_2):
        return -np.sin(x_1) * (np.sin((x_1 ** 2) / np.pi) ** (2 * 10)) -np.sin(x_2) * (np.sin((2 * (x_2 ** 2)) / np.pi) ** (2 * 10))

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_1_bounds, N)


def problem_eight(x_1_candidate = None, x_2_candidate = None, x_1_bounds = [-200, 20], N = 1000):
    """Find the minimun function value for this problem."""
    def f(x_1, x_2):
        return -(x_2 + 47) * (np.sin(np.sqrt(np.abs((x_1 / 2) + (x_2 + 47))))) - x_1 * (np.sin(np.sqrt(np.abs(x_1 - (x_2 + 47)))))

    return generate_values_and_function(f, x_1_candidate, x_2_candidate, x_1_bounds, x_1_bounds, N)


# Old, the "how to use"
# x_1, x_2, y = problem_eight()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(x_1, x_2, y, rstride=10, cstride=10, alpha=0.6, cmap='jet')

# FIXME: Just use x_1_candidate, x_2_candidate and y_candidate to see where it is.
# ax.scatter(x_1, x_2, y, marker='x', s=90, linewidth=3, color='red')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('f(x1, x2)')
#
# plt.tight_layout()
# plt.show()
