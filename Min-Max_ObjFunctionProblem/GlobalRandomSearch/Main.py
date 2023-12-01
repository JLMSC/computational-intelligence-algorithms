import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Any


def GlobalRandomSearch(f, x1_bounds: Tuple[float, float], x2_bounds: Tuple[float, float], max_it: int, dropout: int):
    """Applies Global Random Search in 'f' problem.

    Parameters
    ----------
    f : Callable[[Any, Any], Any]
        A function that'll be minimized or maximized.
    x1_bounds : Tuple[float, float]
        The boundaries of x1.
    x2_bounds : Tuple[float, float]
        The boundaries of x2.
    max_it : int
        The max amount of iterations.
    dropout : int
        The amount of iterations without optimizations to dropout.
    """

    def candidate(xl, xu) -> Tuple[Any, Any]:
        """Generate new candidates based on lower
        and upper bounds.

        Parameters
        ----------
        xl : Any
            The lower bounds.
        xu : Any
            The upper bounds.

        Returns
        -------
        Tuple[Any, Any]
            New generated candidades, x1_cand and x2_cand.
        """
        x1_cand = np.random.uniform(low=xl, high=xu)
        x2_cand = np.random.uniform(low=xl, high=xu)
        return x1_cand, x2_cand

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
    x1_best, x2_best = candidate(xl, xu)
    f_best = f(x1_best, x2_best)

    dropout_count = 0
    for it in range(max_it):
        is_optimized = False

        # Activate dropout.
        if dropout_count > dropout:
            print(f'Dropout activated at {it} iteration.')
            break

        # Generate new candidates.
        x1_cand, x2_cand = candidate(xl, xu)
        F = f(x1_cand, x2_cand)
        # FIXME: > to max, < to min
        if F > f_best:
            # Procreate based on the best candidates from this iteration.
            x1_best, x2_best = x1_cand, x2_cand
            f_best = F

            is_optimized = True

            # Plot the best candidates 'x1' and 'x2' from this iteration.
            ax.scatter(x1_best, x2_best, f_best, marker='x', s=90, linewidth=3, color='red')
        
        if not is_optimized:
            dropout_count += 1
    
    # Show plt.
    plt.show()
