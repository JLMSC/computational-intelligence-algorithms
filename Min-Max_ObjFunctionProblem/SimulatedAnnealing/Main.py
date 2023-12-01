import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Any


def SimulatedAnnealing(f, sigma: float, temperatue: float, x1_bounds: Tuple[float, float], x2_bounds: Tuple[float, float], max_it: int, dropout: int):
    """Applies Simulated Annealing Search in 'f' problem.

    Parameters
    ----------
    f : Callable[[Any, Any], Any]
        A function that'll be minimized or maximized.
    sigma : float
        The value of sigma, should be a value between 0 < sigma < 1.
    temperatue : float
        The initial temperature.
    x1_bounds : Tuple[float, float]
        The boundaries of x1.
    x2_bounds : Tuple[float, float]
        The boundaries fo x2.
    max_it : int
        The max amount of iterations.
    dropout : int
        The amount of iterations without optimizations to dropout.
    """

    def is_within_bounds(x1_cand, x2_cand) -> bool:
        """Check if 'x1_cand' and 'x2_cand' are inside boundaries."""
        return xl <= x1_cand <= xu and xl <= x2_cand <= xu

    # Setup 3d projection of 'f'.
    _x1_fig = np.linspace(x1_bounds[0], x1_bounds[1], 1000)
    _x2_fig = np.linspace(x2_bounds[0], x2_bounds[1], 1000)
    _fig_x1, _fig_x2 = np.meshgrid(_x1_fig, _x2_fig)
    _y_fig = f(_fig_x1, _fig_x2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(_fig_x1, _fig_x2, _y_fig, rstride=10, cstride=10, alpha=0.6, cmap='jet')

    # Both 'x1_best', 'x2_best' and 'f_best' in this stage
    # are just initial values.
    xl, xu = np.min([_x1_fig, _x2_fig]), np.max([_x1_fig, _x2_fig])
    x1_best = np.random.uniform(low=xl, high=xu)
    x2_best = np.random.uniform(low=xl, high=xu)
    f_best = f(x1_best, x2_best)

    dropout_count = 0
    for it in range(max_it):
        is_optimized = False

        # Activate dropout.
        if dropout_count > dropout:
            print(f'Dropout activated at {it} iteration.')
            break

        n1_cand = np.random.normal(loc=0, scale=sigma)
        n2_cand = np.random.normal(loc=0, scale=sigma)
        x1_cand = x1_best + n1_cand
        x2_cand = x2_best + n2_cand

        if not is_within_bounds(x1_cand, x2_cand):
            continue
        
        F = f(x1_cand, x2_cand)
        Pij = np.exp(-((F - f_best) / temperatue))
        # FIXME: > to max, < to min, Pij should not be tampered
        if F < f_best or Pij >= np.random.uniform(0, 1):
            x1_best = x1_cand
            x2_best = x2_cand
            f_best = F

            is_optimized = True

            # Plot the best candidates 'x1' and 'x2' from this iteration.
            ax.scatter(x1_best, x2_best, f_best, marker='x', s=90, linewidth=3, color='red')
        
        # There are three methods to scale temperature, those are:
        temperatue = 0.99 * temperatue
        # temperatue = temperatue / (1 + 0.99 * np.sqrt(temperatue))
        
        if not is_optimized:
            dropout_count += 1
    
    # Show plt.
    plt.show()
