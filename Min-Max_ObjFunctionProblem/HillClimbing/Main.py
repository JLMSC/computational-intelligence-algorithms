import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Any


def HillClimbing(f, epsilon: float, x1_bounds: Tuple[float, float], x2_bounds: Tuple[float, float], max_it: int, max_n: int, dropout: int) -> None:
    """Applies Hill Climbing search in 'f' problem.

    Parameters
    ----------
    f : Callable[[Any, Any], Any]
        A function that'll be minimized or maximized.
    epsilon : float
        The value of epsilon, should be a small value (0, 1).
    x1_bounds : Tuple[float, float]
        The boundaries of x1.
    x2_bounds : Tuple[float, float]
        The boundaries of x2.
    max_it : int
        The max amount of iterations.
    max_n : int
        The max amount of candidates per iteration.
    dropout : int
        The amount of iterations without optimizations to dropout.
    """

    def candidate(x1_best, x2_best) -> Tuple[Any, Any]:
        """Generate new candidates based on current
        iteration best candidates, x1 and x2.

        Parameters
        ----------
        x1_best : Any
            The best candidate x1 from the current
            iteration.
        x2_best : Any
            The best candidate x2 from the current
            iteration.

        Returns
        -------
        Tuple[Any, Any]
            New generated candidades, x1_cand and x2_cand.
        """
        x1_cand = np.random.uniform(low=x1_best - epsilon, high=x1_best + epsilon)
        x2_cand = np.random.uniform(low=x2_best - epsilon, high=x2_best + epsilon)
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
    x1_best, x2_best = 0, 0
    f_best = f(x1_best, x2_best)

    dropout_count = 0
    for it in range(max_it):
        # Activate dropout.
        if dropout_count > dropout:
            print(f'Dropout activated at {it} iteration.')
            break

        for n in range(max_n):
            is_optimized = False

            x1_cand, x2_cand = candidate(x1_best, x2_best)
            F = f(x1_cand, x2_cand)
            # FIXME: > to max, < to min
            if F > f_best:
                # Procreate based on the best candidates from this iteration.
                x1_best, x2_best = x1_cand, x2_cand
                f_best = F

                is_optimized = True

                # Plot the best candidates 'x1' and 'x2' from this iteration.
                ax.scatter(x1_best, x2_best, f_best, marker='x', s=90, linewidth=3, color='red')
                break

            if not is_optimized:
                dropout_count += 1
    
    # Show plt.
    plt.show()
