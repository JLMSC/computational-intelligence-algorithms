from typing import Any, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt


class HillClimbing:
    """Hill Climbing."""

    # TODO: Repeat this R times and doc the most frequent value found.
    @staticmethod
    def search(
        f: Callable[..., Any],
        f_x1_bounds: Tuple[float, float],
        f_x2_bounds: Tuple[float, float],
        f_data_amount: int,
        optimization_method: Callable[..., Any],
        dropout_max_iterations: int,
        max_iterations: int,
        max_candidates: int,
        epsilon: float,
    ) -> None:
        """
        Applies the Hill Climbing algorithm in a specific 'f' problem.

        Tries to minimize/maximize ('optimization_method') the given
        'f' function.

        Parameters
        ----------
        f : Callable[..., Any],
            The function, problem, that will be minimized or maximized.
        f_x1_bounds : int
            Both lower and upper bounds for the random generated x1 values in 'f' function,
        f_x2_bounds : int
            Both lower and upper bounds for the random generated x2 values in 'f' function,
        f_data_amount : int
            The amount of data to be generated using 'f' function.
        optimization_method : Callable[[Any], Any]
            The optimization method that will be used, it can be np.min or np.max.
        dropout_max_iterations : int
            The stop condition when there's no improvements in 'dropout' iterations.
        max_iterations : int
            The maximum amount of iterations.
        max_candidates : int
            The maximum amount of new generated random candidates per iteration.
        epsilon : float
            epsilon, should be a small value between (0, 1).
        """

        def generate_new_candidates(x1, x2) -> tuple:
            """
            Generates new candidates.

            Generates new candidates based on previous iteration/generation
            best 'x1' and 'x2' candidates, also uses 'epsilon'.

            Parameters
            ----------
            x1 : Any
                The previous iteration/generation best candidate.
            x2 : Any
                The previous iteration/generation best candidate.

            Returns
            -------
            tuple
                New candidates based on 'x1' and 'x2' candidates.
            """
            x1 = np.random.uniform(low=x1 - epsilon, high=x1 + epsilon)
            x2 = np.random.uniform(low=x2 - epsilon, high=x2 + epsilon)
            return x1, x2

        # Generate random 'f_data_amount' data between 'f_x1_bounds' and
        # 'f_x2_bounds' using the 'f' function.
        f_x1 = np.linspace(f_x1_bounds[0], f_x1_bounds[1], f_data_amount)
        f_x2 = np.linspace(f_x2_bounds[0], f_x2_bounds[1], f_data_amount)
        f_x1, f_x2 = np.meshgrid(f_x1, f_x2)
        y = f(f_x1, f_x2)

        # Uses the min or max limit of f_x1 and f_x2 based on 'optimization_method'
        # as the starting point.
        x1_candidate = optimization_method(f_x1)
        x2_candidate = optimization_method(f_x2)
        f_candidate = f(x1_candidate, x2_candidate)

        # Setup 3d projection to visualize the current problem.
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(f_x1, f_x2, y, rstride=10, cstride=10, alpha=0.6, cmap="jet")

        # Search, try to optimize the 'f' problem.
        for _ in range(max_iterations):
            # The amount of iterations/generation without optimizations.
            dropout_no_optimizations_count = 0

            # Indicates if there was an optimization in this iteration/generation.
            optimized = False

            # Dropout if no optimization in 'dropout_max_iterations'.
            if dropout_no_optimizations_count >= dropout_max_iterations:
                break

            # Generate 'max_candidates' per iteration,
            # procreating the best candidates.
            for _ in range(max_candidates):
                optimized = False

                new_x1_candidate, new_x2_candidate = generate_new_candidates(
                    x1_candidate, x2_candidate
                )
                F = f(new_x1_candidate, new_x2_candidate)

                # Assuming 'optimization_method' is min or max,
                # this should not (if F < or > f_candidate) equal to
                # f_candidate.
                if optimization_method([F, f_candidate]) != f_candidate:
                    x1_candidate = new_x1_candidate
                    x2_candidate = new_x2_candidate
                    f_candidate = F

                    # Plot the current iteration/generation in the 3d projection.
                    ax.scatter(
                        x1_candidate,
                        x2_candidate,
                        f_candidate,
                        marker="x",
                        s=90,
                        linewidth=3,
                        color="red",
                    )

                    optimized = True
                    break

            # If there's no optimizations in the current iteration/generation
            # sum up the 'dropout_no_optimizations_count'.
            if not optimized:
                dropout_no_optimizations_count += 1

        # Finish 3d projection.
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("f(x1, x2)")

        plt.tight_layout()
        plt.show()

