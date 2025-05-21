import time
from models.tweaks import Tweaks

class LocalSearch:
    @staticmethod
    def local_search(solution, data, time_limit=60.0, max_iterations=1000):
        """
        Perform local search on the given solution using various tweak methods.
        
        Args:
            solution: The initial solution to improve
            data: The problem data
            time_limit: Maximum time to spend on local search in seconds
            max_iterations: Maximum number of iterations to perform
            
        Returns:
            The best solution found during local search
        """
        start_time = time.time()
        best_solution = solution
        iterations = 0
        
        while (time.time() - start_time < time_limit) and (iterations < max_iterations):
            tweak_method = Tweaks.choose_tweak_method()
            new_solution = tweak_method(best_solution, data)
            if new_solution.fitness_score > best_solution.fitness_score:
                best_solution = new_solution
            iterations += 1
        
        return best_solution 