from Utils.Summarizer import ProblemContainer

from HillClimbing.Main import HillClimbing
from LocalRandomSearch.Main import LocalRandomSearch
from GlobalRandomSearch.Main import GlobalRandomSearch
from SimulatedAnnealing.Main import SimulatedAnnealing


if __name__ == '__main__':
    for problem in ProblemContainer.values():
        HillClimbing(
            problem.f,
            0.2,
            problem.x1_bounds,
            problem.x2_bounds,
            1000,
            1000,
            dropout=1000,
        )

        # LocalRandomSearch(
        #     f=problem.f,
        #     sigma=0.1,
        #     x1_bounds=problem.x1_bounds,
        #     x2_bounds=problem.x2_bounds,
        #     max_it=1000,
        #     dropout=1000,
        # )

        # GlobalRandomSearch(
        #     f=problem.f,
        #     x1_bounds=problem.x1_bounds,
        #     x2_bounds=problem.x2_bounds,
        #     max_it=1000,
        #     dropout=1000,
        # )

        # SimulatedAnnealing(
        #     f=problem.f,
        #     sigma=0.2,
        #     temperatue=1.0,
        #     x1_bounds=problem.x1_bounds,
        #     x2_bounds=problem.x2_bounds,
        #     max_it=1000,
        #     dropout=1000,
        # )