import numpy as np


# TODO: Add docs for every function.
class HillClimbing:
    """Hill Climbing"""

    @staticmethod
    def fit(obj_function, epsilon: float = 0.1, max_iter: int = 100):
        """TODO: Doc"""

        def candidate(x_1, x_2):
            x_1_candidate = x_1 + np.random.uniform(-epsilon, epsilon)
            x_2_candidate = x_2 + np.random.uniform(-epsilon, epsilon)
            return x_1_candidate, x_2_candidate

        # todo: Make a func to visualize every iteration?
        # todo: Remove later, just a visualization.
        import matplotlib.pyplot as plt

        x_1, x_2, y = obj_function()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x_1, x_2, y, rstride=10, cstride=10, alpha=0.6, cmap='jet')

        # warn: epsilon must be between (0, 1)
        # todo: find a good value for epsilon to get the optimal solution.

        # fixme: 0, 0 or min/max limit
        x_1_candidate, x_2_candidate = 0, 0
        f_best = obj_function(x_1_candidate=x_1_candidate, x_2_candidate=x_2_candidate)
        max_candidates = 10
        for _ in range(max_iter):
            for _ in range(max_candidates):
                y = candidate(x_1_candidate, x_2_candidate)
                F = obj_function(x_1_candidate=y[0], x_2_candidate=y[1])
                if F > f_best:
                    x_1_candidate, x_2_candidate = y
                    f_best = F

                    # todo: the scatter should be added here with gray color or something
                    # (to visualize each iteration)
                    break
 
        # todo: Remove later, just a visualization.
        ax.scatter(x_1_candidate, x_2_candidate, f_best[-1], marker='x', s=90, linewidth=3, color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('f(x1, x2)')

        plt.tight_layout()
        plt.show()       

        # WARN: no returns are needed.
        return f_best

