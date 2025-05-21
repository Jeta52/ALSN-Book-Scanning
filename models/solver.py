import random
import time
from models.solution import Solution
from models.initial_solution import InitialSolution
from models.local_search import LocalSearch

class Solver:
    def iterated_local_search(self, data, time_limit=300, max_iterations=1000, pool_size=5):
        """
        Perform Iterated Local Search (ILS) on the given problem data with enhanced acceptance and home base selection.
        Args:
            data: The problem data (libraries, scores, num_days, etc.)
            time_limit: Maximum time to run the algorithm in seconds
            max_iterations: Maximum number of iterations to perform
            pool_size: Number of recent local optima to keep in the homebase pool
        Returns:
            The best solution found during the search
        """
        
        # Detect instance size
        num_libraries = len(data.libs)
        num_books = len(data.scores)
        instance_size = num_libraries * num_books
        is_small_instance = instance_size < 100000
        
        # Set parameters based on instance size
        if is_small_instance:
            print(f"Small instance detected: {num_libraries} libraries, {num_books} books")
            # For small instances, use more refined local search
            max_local_search_time = 3.0  # Longer local search time for small instances
        else:
            max_local_search_time = 1.0  # Default local search time
        
        current_solution = InitialSolution.generate_initial_solution(data)
        start_time = time.time()
        best_solution = current_solution
        homebase_pool = []
        
        stagnation_counter = 0
        max_stagnation = 50

        total_iterations = 0
     
        homebase_pool.append(current_solution)
        print(f"Initial solution fitness: {current_solution.fitness_score}")

        iteration = 0
        while time.time() - start_time < time_limit and iteration < max_iterations:
            remaining_time = time_limit - (time.time() - start_time)
            progress = iteration / max_iterations
            
            # Adjust local search time based on instance size and progress
            if is_small_instance:
                # Allocate more time for local search on small instances
                local_search_time = min(max_local_search_time, max(0.2, remaining_time * (1 - progress) / 60))
            else:
                local_search_time = min(max_local_search_time, max(0.1, remaining_time * (1 - progress) / 100))
            
            # Strategy selection - adapt based on stagnation
            if stagnation_counter > max_stagnation * 0.7:
                # When stagnating, use more disruptive perturbation
                perturbation_strategies = ['remove_insert', 'reorder', 'shuffle']
                weights = [0.5, 0.3, 0.2]  # Prefer remove_insert when stagnating
            else:
                perturbation_strategies = ['remove_insert', 'reorder', 'shuffle']
                weights = [0.4, 0.4, 0.2]  # Balanced normally
                
            # For small instances, adjust strategy weights
            if is_small_instance:
                if stagnation_counter > max_stagnation * 0.7:
                    # When stagnating on small instances, use more focused perturbation
                    weights = [0.6, 0.3, 0.1]  # Even more emphasis on remove_insert
                else:
                    weights = [0.5, 0.4, 0.1]  # Favor library reordering for small instances
                    
            perturbation_strategy = random.choices(perturbation_strategies, weights=weights, k=1)[0]
            
            # Apply perturbation with adaptations for small instances
            perturbed_solution = self.perturb_solution(
                current_solution, 
                data, 
                strategy=perturbation_strategy,
                stagnation_level=stagnation_counter/max_stagnation,
                is_small_instance=is_small_instance
            )
            
            if perturbed_solution.fitness_score > best_solution.fitness_score:
                print(f"New best solution found after perturbation: {perturbed_solution.fitness_score}")
         
            # For small instances, use more intensive local search
            if is_small_instance:
                max_iterations_ls = 2000  # Double iterations for small instances
            else:
                max_iterations_ls = 1000  # Default iterations
                
            improved_solution = LocalSearch.local_search(
                perturbed_solution, 
                data, 
                time_limit=local_search_time,
                max_iterations=max_iterations_ls
            )

            accept = False
            if improved_solution.fitness_score > current_solution.fitness_score:
                accept = True
                stagnation_counter = 0
            else:
                quality_diff = (current_solution.fitness_score - improved_solution.fitness_score) / current_solution.fitness_score
                
                # For small instances, be more selective with acceptance
                if is_small_instance:
                    # Lower acceptance probability for worse solutions on small instances
                    accept_prob = 0.1 * (1 - quality_diff) * (1 + stagnation_counter/max_stagnation)
                else:
                    accept_prob = 0.2 * (1 - quality_diff)
                    
                if random.random() < accept_prob:
                    accept = True

            if accept:
                current_solution = improved_solution
                if all(s.fitness_score != current_solution.fitness_score for s in homebase_pool):
                    homebase_pool.append(current_solution)
                    if len(homebase_pool) > pool_size:
                        homebase_pool.sort(key=lambda x: x.fitness_score)
                        homebase_pool.pop(0)
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution
                    print(f"New best solution found: {best_solution.fitness_score}")

            stagnation_counter += 1
            if stagnation_counter >= max_stagnation:
                print(f"Stagnation detected after {stagnation_counter} iterations. Restarting...")
                current_solution = random.choice(homebase_pool)
                stagnation_counter = 0

            if homebase_pool:
                weights = [s.fitness_score for s in homebase_pool]
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w/total_weight for w in weights]
                    current_solution = random.choices(homebase_pool, weights=weights, k=1)[0]
                else:
                    current_solution = random.choice(homebase_pool)

            iteration += 1
            total_iterations += 1
            
            # For small instances, apply additional local search periodically
            if is_small_instance and iteration % 10 == 0:
                # Every 10 iterations, do an extra intensive local search
                extra_time = min(3.0, local_search_time * 2)
                current_solution = LocalSearch.local_search(
                    current_solution,
                    data,
                    time_limit=extra_time,
                    max_iterations=2500
                )
                if current_solution.fitness_score > best_solution.fitness_score:
                    best_solution = current_solution
                    print(f"New best solution found during extra local search: {best_solution.fitness_score}")

        total_time = time.time() - start_time
        print(f"\nILS finished after {total_iterations} iterations and {total_time:.2f} seconds.")
        print(f"Final best score: {best_solution.fitness_score}")
        return best_solution
        
    def perturb_solution(self, solution, data, strategy='remove_insert', stagnation_level=0.0, is_small_instance=False):
        """
        Perturb the current solution using various strategies with adaptations for small instances.
        Args:
            solution: The current solution to perturb
            data: The problem data
            strategy: The perturbation strategy to use
            stagnation_level: Level of stagnation (0.0-1.0)
            is_small_instance: Whether this is a small problem instance
        Returns:
            A new perturbed solution
        """
        new_solution = self._clone_solution(solution)
        
        if strategy == 'remove_insert':
            return self._perturb_remove_insert(new_solution, data, stagnation_level, is_small_instance)
        elif strategy == 'reorder':
            return self._perturb_reorder(new_solution, data, stagnation_level, is_small_instance)
        elif strategy == 'shuffle':
            return self._perturb_shuffle(new_solution, data, stagnation_level, is_small_instance)
        else:
            return self._perturb_remove_insert(new_solution, data, stagnation_level, is_small_instance)
            
    def _clone_solution(self, solution):
        return Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )
    
    def _calculate_library_efficiency(self, library, data, scanned_books):
        """
        Calculate library efficiency based on potential score, unique books, and signup cost.
        """
        # Get unscanned books
        available_books = {book.id for book in library.books} - scanned_books
        if not available_books:
            return 0
            
        # Calculate score potential
        score_potential = sum(data.scores[book_id] for book_id in available_books)
        
        # Calculate unique ratio
        unique_ratio = len(available_books) / len(library.books)
        
        # Calculate time efficiency
        time_efficiency = library.books_per_day / max(1, library.signup_days)
        
        # Combined score
        efficiency = (score_potential * 0.6 + 
                     unique_ratio * 0.2 + 
                     time_efficiency * 0.2)
                     
        return efficiency
        
    def _perturb_remove_insert(self, solution, data, stagnation_level=0.0, is_small_instance=False):
        # Adaptive perturbation size based on stagnation level
        base_size = len(solution.signed_libraries) // 10  # 10% of libraries
        
        if is_small_instance:
            # For small instances, start with smaller perturbations
            if stagnation_level < 0.3:
                # Low stagnation: perturb 5-10% of libraries
                num_to_perturb = max(1, int(len(solution.signed_libraries) * 0.05 * (1 + stagnation_level)))
            elif stagnation_level < 0.7:
                # Medium stagnation: perturb 10-15% of libraries
                num_to_perturb = max(1, int(len(solution.signed_libraries) * 0.10 * (1 + stagnation_level / 2)))
            else:
                # High stagnation: perturb 15-20% of libraries
                num_to_perturb = max(1, int(len(solution.signed_libraries) * 0.15 * (1 + stagnation_level / 3)))
                
            # Cap at 20% for small instances
            num_to_perturb = min(num_to_perturb, int(len(solution.signed_libraries) * 0.20))
        else:
            # For larger instances, use standard sizing with stagnation adaptation
            num_to_perturb = max(1, min(5, int(base_size * (1 + stagnation_level))))
        
        # Select libraries to remove - for small instances, be more strategic
        if is_small_instance and random.random() < 0.7:  # 70% chance for strategic selection
            # For small instances, use biased selection to favor:
            # 1. Libraries with low efficiency in current position
            # 2. Libraries that might perform better elsewhere
            
            # Calculate efficiency scores for signed libraries
            library_scores = []
            for i, lib_id in enumerate(solution.signed_libraries):
                library = data.libs[lib_id]
                efficiency = self._calculate_library_efficiency(library, data, solution.scanned_books)
                library_scores.append((i, lib_id, efficiency))
            
            # Sort by efficiency (ascending so lowest scores come first)
            library_scores.sort(key=lambda x: x[2])
            
            # Select indices of lowest-scoring libraries to remove
            to_remove_indices = sorted([score[0] for score in library_scores[:num_to_perturb]])
        else:
            # Standard random selection
            indices = list(range(len(solution.signed_libraries)))
            to_remove_indices = sorted(random.sample(indices, num_to_perturb))
        
        # Remove selected libraries
        destroyed_indices = []
        for offset, idx in enumerate(to_remove_indices):
            real_idx = idx - offset
            lib_id = solution.signed_libraries.pop(real_idx)
            destroyed_indices.append(idx)
            solution.unsigned_libraries.append(lib_id)
            if lib_id in solution.scanned_books_per_library:
                del solution.scanned_books_per_library[lib_id]
        
        # For small instances, use intelligent insertion 50% of the time
        if is_small_instance and random.random() < 0.5:
            # Calculate efficiency for unsigned libraries
            library_scores = []
            for lib_id in solution.unsigned_libraries:
                library = data.libs[lib_id]
                efficiency = self._calculate_library_efficiency(library, data, solution.scanned_books)
                library_scores.append((lib_id, efficiency))
            
            # Sort by efficiency (descending)
            library_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-scoring libraries
            top_libs = [lib_id for lib_id, _ in library_scores[:num_to_perturb]]
            
            # Remove these from unsigned_libraries
            for lib_id in top_libs:
                solution.unsigned_libraries.remove(lib_id)
            
            # Insert strategically
            for idx, lib_id in zip(destroyed_indices, top_libs):
                insert_idx = min(idx, len(solution.signed_libraries))
                solution.signed_libraries.insert(insert_idx, lib_id)
        else:
            # Standard random approach
            random.shuffle(solution.unsigned_libraries)
            for idx in destroyed_indices:
                if not solution.unsigned_libraries:
                    break
                lib_id = solution.unsigned_libraries.pop()
                insert_idx = min(idx, len(solution.signed_libraries))
                solution.signed_libraries.insert(insert_idx, lib_id)
            
        return self._rebuild_solution(solution, data)
        
    def _perturb_reorder(self, solution, data, stagnation_level=0.0, is_small_instance=False):
        if len(solution.signed_libraries) < 2:
            return solution
            
        # Adaptive perturbation size based on stagnation level and instance size
        if is_small_instance:
            # For small instances, reorder more libraries
            if stagnation_level < 0.3:
                # Low stagnation: reorder 10-15% of libraries
                num_to_reorder = max(2, int(len(solution.signed_libraries) * 0.10 * (1 + stagnation_level / 2)))
            elif stagnation_level < 0.7:
                # Medium stagnation: reorder 15-20% of libraries
                num_to_reorder = max(2, int(len(solution.signed_libraries) * 0.15 * (1 + stagnation_level / 3)))
            else:
                # High stagnation: reorder 20-25% of libraries
                num_to_reorder = max(2, int(len(solution.signed_libraries) * 0.20 * (1 + stagnation_level / 4)))
                
            # Cap at 25% for small instances
            num_to_reorder = min(num_to_reorder, int(len(solution.signed_libraries) * 0.25))
        else:
            # For larger instances, use standard approach
            num_to_reorder = min(5, max(2, int(len(solution.signed_libraries) // 3 * (1 + stagnation_level / 2))))
        
        indices = random.sample(range(len(solution.signed_libraries)), num_to_reorder)
        
        # For small instances, use intelligent reordering occasionally
        if is_small_instance and random.random() < 0.4:  # 40% chance for intelligent reordering
            # Calculate efficiency for each library to be reordered
            library_scores = []
            for i in indices:
                lib_id = solution.signed_libraries[i]
                library = data.libs[lib_id]
                efficiency = self._calculate_library_efficiency(library, data, solution.scanned_books)
                library_scores.append((i, lib_id, efficiency))
            
            # Sort by efficiency (descending)
            library_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Assign libraries to positions based on efficiency
            # Higher efficiency libraries go to earlier positions
            sorted_indices = sorted(indices)
            for i, (_, lib_id, _) in enumerate(library_scores):
                pos = sorted_indices[i]
                solution.signed_libraries[pos] = lib_id
        else:
            # Standard random reordering
            libraries_to_reorder = [solution.signed_libraries[i] for i in indices]
            random.shuffle(libraries_to_reorder)
            
            for i, lib_id in zip(indices, libraries_to_reorder):
                solution.signed_libraries[i] = lib_id
            
        return self._rebuild_solution(solution, data)
        
    def _perturb_shuffle(self, solution, data, stagnation_level=0.0, is_small_instance=False):
        if len(solution.signed_libraries) < 2:
            return solution
            
        # Adaptive segment size based on stagnation level and instance size
        if is_small_instance:
            # For small instances, use larger segments
            segment_size_min = max(2, int(len(solution.signed_libraries) * 0.10))
            segment_size_max = max(3, int(len(solution.signed_libraries) * (0.20 + stagnation_level * 0.10)))
            
            # Ensure max doesn't exceed array length
            segment_size_max = min(segment_size_max, len(solution.signed_libraries))
            
            # Random segment size within range
            segment_size = random.randint(segment_size_min, segment_size_max)
            
            # Select start position to ensure we don't exceed array bounds
            max_start = len(solution.signed_libraries) - segment_size
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + segment_size
        else:
            # Standard approach for larger instances
            start_idx = random.randint(0, len(solution.signed_libraries) - 2)
            end_idx = random.randint(start_idx + 1, len(solution.signed_libraries))
        
        # For small instances, occasionally use intelligent shuffling
        if is_small_instance and random.random() < 0.3:  # 30% chance for intelligent shuffling
            # Extract segment to shuffle
            segment = solution.signed_libraries[start_idx:end_idx]
            
            # Calculate efficiency for each library in segment
            library_scores = []
            for lib_id in segment:
                library = data.libs[lib_id]
                efficiency = self._calculate_library_efficiency(library, data, solution.scanned_books)
                library_scores.append((lib_id, efficiency))
            
            # Sort by efficiency (descending)
            library_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Replace segment with efficiency-sorted libraries
            solution.signed_libraries[start_idx:end_idx] = [lib_id for lib_id, _ in library_scores]
        else:
            # Standard random shuffling
            subsegment = solution.signed_libraries[start_idx:end_idx]
            random.shuffle(subsegment)
            solution.signed_libraries[start_idx:end_idx] = subsegment
        
        return self._rebuild_solution(solution, data)
        
    def _rebuild_solution(self, solution, data):
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        
        for lib_id in solution.signed_libraries:
            library = data.libs[lib_id]
            if curr_time + library.signup_days >= data.num_days:
                continue
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            if available_books:
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
                
        solution.scanned_books_per_library = new_scanned_books_per_library
        solution.scanned_books = new_scanned_books
        solution.calculate_fitness_score(data.scores)
        return solution
