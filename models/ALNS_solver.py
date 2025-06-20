import random
import time
import math
from typing import List, Set, Dict, Tuple
from models.solution import Solution
from models.initial_solution import InitialSolution
from models.library import Library


class ALNSSolver:
    def __init__(self, data, destroy_percentage=0.2):
        self.data = data
        self.destroy_percentage = destroy_percentage
        self.min_destroy_percentage = 0.1
        self.max_destroy_percentage = 0.4
        self.iterations_since_improvement = 0
        
        # Reaction factor for weight adjustments (how quickly weights adapt)
        self.reaction_factor = 0.1
        
        # Segment size for weight adjustments
        self.segment_size = 50  # Reduced for more frequent updates
        
        # Score weights for different improvement types
        self.score_weights = {
            'global_best': 20,    # New best solution found
            'significant': 10,    # Improvement > 1% over current
            'improvement': 6,     # Any improvement over current
            'accepted': 3,        # Accepted but not better
            'rejected': 1         # Not accepted (still gets small reward)
        }
        
        # Weights for destroy operators
        self.destroy_weights = {
            self.random_library_removal: 1.0,
            self.low_efficiency_library_removal: 1.0,
            self.duplicate_book_removal: 1.0,
            self.overlap_based_removal: 1.0
        }
        
        # Weights for repair operators
        self.repair_weights = {
            self.greedy_max_unique_score_repair: 1.0,
            self.regret_score_repair: 1.0,
            self.max_books_repair: 1.0,
            self.random_feasible_repair: 1.0
        }
        
        # Score tracking for adaptive weight adjustment
        self.destroy_scores = {op: 0 for op in self.destroy_weights}
        self.repair_scores = {op: 0 for op in self.repair_weights}
        
        # Usage tracking
        self.destroy_usage = {op: 0 for op in self.destroy_weights}
        self.repair_usage = {op: 0 for op in self.repair_weights}
        
        # Statistics tracking
        self.stats = {
            'iterations': 0,
            'improvements': 0,
            'time_to_best': 0,
            'initial_score': 0,
            'final_score': 0,
            'best_destroy_op': None,
            'best_repair_op': None,
            'accepted_worse': 0,
            'rejected': 0
        }
        
    def solve(self, time_limit=600) -> Solution:  # 10 minutes = 600 seconds
        """
        Main ALNS solving method that runs for 10 minutes per instance
        """
        start_time = time.time()
        last_print_time = start_time
        print_interval = 30  # Print progress every 30 seconds
        
        # Get initial solution using GRASP
        print("Generating initial solution using GRASP...")
        current_solution = InitialSolution.generate_initial_solution_grasp(self.data)
        best_solution = current_solution.clone()
        # Ensure best_solution has correct fitness score
        best_solution.calculate_fitness_score(self.data.scores)
        
        self.stats['initial_score'] = current_solution.fitness_score
        print(f"Initial solution score: {current_solution.fitness_score}")
        
        # Track best operators
        best_destroy_op_count = {op: 0 for op in self.destroy_weights}
        best_repair_op_count = {op: 0 for op in self.repair_weights}
        
        # Temperature for simulated annealing
        initial_temp = current_solution.fitness_score * 0.1  # 10% of initial score
        
        while (time.time() - start_time) < time_limit:
            # Select destroy and repair operators based on weights
            destroy_op = self._select_operator(self.destroy_weights)
            repair_op = self._select_operator(self.repair_weights)
            
            # Create a copy of current solution
            temp_solution = current_solution.clone()
            # Ensure cloned solution has correct fitness score
            temp_solution.calculate_fitness_score(self.data.scores)
            
            # Apply destroy operator
            removed_libraries = destroy_op(temp_solution)
            
            # Apply repair operator
            new_solution = repair_op(temp_solution, removed_libraries)
            
            # Calculate improvement
            delta = new_solution.fitness_score - current_solution.fitness_score
            
            # Accept or reject the new solution - ONLY ACCEPT BETTER SOLUTIONS
            accept = False
            if delta > 0:  # Only accept strictly better solutions
                accept = True
                current_solution = new_solution
                self.stats['improvements'] += 1
                
                # Update scores for successful operators
                if new_solution.fitness_score > best_solution.fitness_score:
                    # Global best improvement
                    self.destroy_scores[destroy_op] += self.score_weights['global_best']
                    self.repair_scores[repair_op] += self.score_weights['global_best']
                    
                    best_solution = new_solution.clone()
                    # Ensure best_solution has correct fitness score
                    best_solution.calculate_fitness_score(self.data.scores)
                    
                    self.stats['time_to_best'] = time.time() - start_time
                    best_destroy_op_count[destroy_op] += 1
                    best_repair_op_count[repair_op] += 1
                    
                    print(f"New best solution found: {best_solution.fitness_score} (improvement: {delta})")
                    
                    # Reset iterations since improvement and reduce destroy percentage
                    self.iterations_since_improvement = 0
                    self.destroy_percentage = max(self.min_destroy_percentage, 
                                                self.destroy_percentage * 0.95)
                else:
                    # Regular improvement
                    improvement_percentage = delta / current_solution.fitness_score
                    if improvement_percentage > 0.01:  # More than 1% improvement
                        self.destroy_scores[destroy_op] += self.score_weights['significant']
                        self.repair_scores[repair_op] += self.score_weights['significant']
                    else:
                        self.destroy_scores[destroy_op] += self.score_weights['improvement']
                        self.repair_scores[repair_op] += self.score_weights['improvement']
            else:
                # Solution is worse or equal - reject it
                self.stats['rejected'] += 1
                self.destroy_scores[destroy_op] += self.score_weights['rejected']
                self.repair_scores[repair_op] += self.score_weights['rejected']
            
            # Increase destroy percentage if stuck in local optima
            self.iterations_since_improvement += 1
            if self.iterations_since_improvement > 100:
                self.destroy_percentage = min(self.max_destroy_percentage, 
                                            self.destroy_percentage * 1.05)
                self.iterations_since_improvement = 0
            
            # Update weights periodically
            if self.stats['iterations'] > 0 and self.stats['iterations'] % self.segment_size == 0:
                self._update_weights()
            
            self.stats['iterations'] += 1
            
            # Print progress every 30 seconds
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                elapsed_time = current_time - start_time
                print(f"\nProgress after {elapsed_time:.1f} seconds:")
                print(f"Iterations: {self.stats['iterations']}")
                print(f"Current score: {current_solution.fitness_score}")
                print(f"Best score: {best_solution.fitness_score}")
                print(f"Improvements: {self.stats['improvements']}")
                print(f"Rejected: {self.stats['rejected']}")
                print(f"Destroy percentage: {self.destroy_percentage:.3f}")
                
                # Print top operators
                top_destroy = sorted(self.destroy_weights.items(), key=lambda x: x[1], reverse=True)[:2]
                top_repair = sorted(self.repair_weights.items(), key=lambda x: x[1], reverse=True)[:2]
                print(f"Top destroy operators: {top_destroy[0][0].__name__} ({top_destroy[0][1]:.2f}), {top_destroy[1][0].__name__} ({top_destroy[1][1]:.2f})")
                print(f"Top repair operators: {top_repair[0][0].__name__} ({top_repair[0][1]:.2f}), {top_repair[1][0].__name__} ({top_repair[1][1]:.2f})")
                
                last_print_time = current_time
        
        # Final statistics
        self.stats['final_score'] = best_solution.fitness_score
        
        # Determine best operators
        if best_destroy_op_count:
            self.stats['best_destroy_op'] = max(best_destroy_op_count.items(), key=lambda x: x[1])[0].__name__
        if best_repair_op_count:
            self.stats['best_repair_op'] = max(best_repair_op_count.items(), key=lambda x: x[1])[0].__name__
        
        total_time = time.time() - start_time
        print("\nFinal Statistics:")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Total iterations: {self.stats['iterations']}")
        print(f"Initial score: {self.stats['initial_score']}")
        print(f"Final score: {self.stats['final_score']}")
        print(f"Improvement: {self.stats['final_score'] - self.stats['initial_score']}")
        print(f"Time to best solution: {self.stats['time_to_best']:.1f} seconds")
        if self.stats['best_destroy_op']:
            print(f"Most successful destroy operator: {self.stats['best_destroy_op']}")
        if self.stats['best_repair_op']:
            print(f"Most successful repair operator: {self.stats['best_repair_op']}")
        
        return best_solution
    
    def _select_operator(self, weights: Dict) -> callable:
        """Select an operator based on weights with roulette wheel selection"""
        operators = list(weights.keys())
        weights_list = [max(0.1, weights[op]) for op in operators]  # Ensure minimum weight
        
        # Update usage counter for selected operator
        selected = random.choices(operators, weights=weights_list, k=1)[0]
        if selected in self.destroy_weights:
            self.destroy_usage[selected] += 1
        else:
            self.repair_usage[selected] += 1
            
        return selected
    
    def _update_weights(self):
        """Update weights of operators based on their success using reaction factor"""
        # Update destroy weights
        for op in self.destroy_weights:
            if self.destroy_usage[op] > 0:  # Only update if operator was used
                avg_score = self.destroy_scores[op] / self.destroy_usage[op]
                # Normalize score to [0, 1] range
                normalized_score = avg_score / max(1, max(self.score_weights.values()))
                
                # Update weight using reaction factor
                self.destroy_weights[op] = (
                    (1 - self.reaction_factor) * self.destroy_weights[op] +
                    self.reaction_factor * max(0.1, normalized_score)
                )
            
        # Update repair weights similarly
        for op in self.repair_weights:
            if self.repair_usage[op] > 0:  # Only update if operator was used
                avg_score = self.repair_scores[op] / self.repair_usage[op]
                # Normalize score to [0, 1] range
                normalized_score = avg_score / max(1, max(self.score_weights.values()))
                
                # Update weight using reaction factor
                self.repair_weights[op] = (
                    (1 - self.reaction_factor) * self.repair_weights[op] +
                    self.reaction_factor * max(0.1, normalized_score)
                )
            
        # Reset scores and usage for next segment
        self.destroy_scores = {op: 0 for op in self.destroy_weights}
        self.repair_scores = {op: 0 for op in self.repair_weights}
        self.destroy_usage = {op: 0 for op in self.destroy_weights}
        self.repair_usage = {op: 0 for op in self.repair_weights}
    
    # Destroy Operators
    def random_library_removal(self, solution: Solution) -> List[int]:
        """Randomly remove libraries from the solution"""
        if not solution.signed_libraries:
            return []
            
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        num_to_remove = min(num_to_remove, len(solution.signed_libraries))
            
        indices_to_remove = random.sample(range(len(solution.signed_libraries)), num_to_remove)
        removed_libraries = []
        
        for idx in sorted(indices_to_remove, reverse=True):
            lib_id = solution.signed_libraries.pop(idx)
            removed_libraries.append(lib_id)
            solution.unsigned_libraries.append(lib_id)
            
            # Remove books from the removed library
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                solution.scanned_books.difference_update(books)
                del solution.scanned_books_per_library[lib_id]
        
        return removed_libraries
    
    def low_efficiency_library_removal(self, solution: Solution) -> List[int]:
        """Remove libraries with lowest efficiency (score/signup days)"""
        if not solution.signed_libraries:
            return []
            
        library_efficiencies = []
        for lib_id in solution.signed_libraries:
            library = self.data.libs[lib_id]
            books = solution.scanned_books_per_library.get(lib_id, [])
            score = sum(self.data.scores[book_id] for book_id in books)
            efficiency = score / max(1, library.signup_days)  # Avoid division by zero
            library_efficiencies.append((lib_id, efficiency))
            
        library_efficiencies.sort(key=lambda x: x[1])
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        num_to_remove = min(num_to_remove, len(library_efficiencies))
            
        removed_libraries = []
        for i in range(num_to_remove):
            lib_id = library_efficiencies[i][0]
            solution.signed_libraries.remove(lib_id)
            solution.unsigned_libraries.append(lib_id)
            removed_libraries.append(lib_id)
            
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                solution.scanned_books.difference_update(books)
                del solution.scanned_books_per_library[lib_id]
                
        return removed_libraries
    
    def duplicate_book_removal(self, solution: Solution) -> List[int]:
        """Remove libraries with highest number of duplicate books"""
        if not solution.signed_libraries:
            return []
            
        library_duplicates = []
        for lib_id in solution.signed_libraries:
            books = solution.scanned_books_per_library.get(lib_id, [])
            duplicates = 0
            for book_id in books:
                # Count how many other libraries also have this book
                for other_lib_id, other_books in solution.scanned_books_per_library.items():
                    if other_lib_id != lib_id and book_id in other_books:
                        duplicates += 1
            library_duplicates.append((lib_id, duplicates))
            
        library_duplicates.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        num_to_remove = min(num_to_remove, len(library_duplicates))
            
        removed_libraries = []
        for i in range(num_to_remove):
            lib_id = library_duplicates[i][0]
            solution.signed_libraries.remove(lib_id)
            solution.unsigned_libraries.append(lib_id)
            removed_libraries.append(lib_id)
            
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                solution.scanned_books.difference_update(books)
                del solution.scanned_books_per_library[lib_id]
                
        return removed_libraries
    
    def overlap_based_removal(self, solution: Solution) -> List[int]:
        """Remove libraries with highest overlap in book coverage"""
        if not solution.signed_libraries:
            return []
            
        library_overlaps = []
        for lib_id in solution.signed_libraries:
            books = set(solution.scanned_books_per_library.get(lib_id, []))
            overlap = 0
            for other_lib_id in solution.signed_libraries:
                if other_lib_id != lib_id:
                    other_books = set(solution.scanned_books_per_library.get(other_lib_id, []))
                    overlap += len(books.intersection(other_books))
            library_overlaps.append((lib_id, overlap))
            
        library_overlaps.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        num_to_remove = min(num_to_remove, len(library_overlaps))
            
        removed_libraries = []
        for i in range(num_to_remove):
            lib_id = library_overlaps[i][0]
            solution.signed_libraries.remove(lib_id)
            solution.unsigned_libraries.append(lib_id)
            removed_libraries.append(lib_id)
            
            if lib_id in solution.scanned_books_per_library:
                books = solution.scanned_books_per_library[lib_id]
                solution.scanned_books.difference_update(books)
                del solution.scanned_books_per_library[lib_id]
                
        return removed_libraries
    
    # Repair Operators
    def greedy_max_unique_score_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by adding libraries that maximize unique book scores"""
        if not removed_libraries:
            solution.calculate_fitness_score(self.data.scores)
            return solution
            
        # Calculate current time used
        curr_time = 0
        for lib_id in solution.signed_libraries:
            curr_time += self.data.libs[lib_id].signup_days
        
        # Sort removed libraries by potential score
        library_potentials = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            # Get unique books and their scores
            unique_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            unique_books.sort(key=lambda b: self.data.scores[b], reverse=True)
            unique_books = unique_books[:max_books]
            
            if unique_books:
                potential_score = sum(self.data.scores[b] for b in unique_books)
                library_potentials.append((potential_score, lib_id, unique_books))
        
        # Sort by potential score (descending)
        library_potentials.sort(key=lambda x: x[0], reverse=True)
        
        # Add libraries in order of potential
        for potential_score, lib_id, books in library_potentials:
            library = self.data.libs[lib_id]
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            solution.signed_libraries.append(lib_id)
            solution.unsigned_libraries.remove(lib_id)
            solution.scanned_books_per_library[lib_id] = books
            solution.scanned_books.update(books)
            curr_time += library.signup_days
                
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def regret_score_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution using regret-based insertion"""
        if not removed_libraries:
            solution.calculate_fitness_score(self.data.scores)
            return solution
            
        # Calculate current time used
        curr_time = 0
        for lib_id in solution.signed_libraries:
            curr_time += self.data.libs[lib_id].signup_days
        
        remaining_libraries = removed_libraries.copy()
        
        while remaining_libraries:
            best_regret = -1
            best_lib_id = None
            best_books = None
            best_time_cost = 0
            
            for lib_id in remaining_libraries:
                library = self.data.libs[lib_id]
                if curr_time + library.signup_days >= self.data.num_days:
                    continue
                    
                time_left = self.data.num_days - (curr_time + library.signup_days)
                max_books = time_left * library.books_per_day
                
                # Calculate regret (score that would be lost if not chosen now)
                available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
                if not available_books:
                    continue
                    
                sorted_books = sorted(available_books, key=lambda b: self.data.scores[b], reverse=True)[:max_books]
                regret_score = sum(self.data.scores[b] for b in sorted_books)
                
                if regret_score > best_regret:
                    best_regret = regret_score
                    best_lib_id = lib_id
                    best_books = sorted_books
                    best_time_cost = library.signup_days
            
            if best_lib_id is None:
                break
                
            remaining_libraries.remove(best_lib_id)
            solution.signed_libraries.append(best_lib_id)
            solution.unsigned_libraries.remove(best_lib_id)
            solution.scanned_books_per_library[best_lib_id] = best_books
            solution.scanned_books.update(best_books)
            curr_time += best_time_cost
            
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def max_books_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by maximizing the number of books that can be scanned"""
        if not removed_libraries:
            solution.calculate_fitness_score(self.data.scores)
            return solution
            
        # Calculate current time used
        curr_time = 0
        for lib_id in solution.signed_libraries:
            curr_time += self.data.libs[lib_id].signup_days
        
        library_potentials = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            books_to_scan = available_books[:max_books]
            
            if books_to_scan:
                library_potentials.append((len(books_to_scan), lib_id, books_to_scan))
                
        # Sort by number of books (descending)
        library_potentials.sort(key=lambda x: x[0], reverse=True)
        
        for num_books, lib_id, books in library_potentials:
            library = self.data.libs[lib_id]
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            solution.signed_libraries.append(lib_id)
            solution.unsigned_libraries.remove(lib_id)
            solution.scanned_books_per_library[lib_id] = books
            solution.scanned_books.update(books)
            curr_time += library.signup_days
            
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def random_feasible_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by randomly inserting libraries while maintaining feasibility"""
        if not removed_libraries:
            solution.calculate_fitness_score(self.data.scores)
            return solution
            
        # Calculate current time used
        curr_time = 0
        for lib_id in solution.signed_libraries:
            curr_time += self.data.libs[lib_id].signup_days
        
        random.shuffle(removed_libraries)
        
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            books_to_scan = available_books[:max_books]
            
            if books_to_scan:
                solution.signed_libraries.append(lib_id)
                solution.unsigned_libraries.remove(lib_id)
                solution.scanned_books_per_library[lib_id] = books_to_scan
                solution.scanned_books.update(books_to_scan)
                curr_time += library.signup_days
                
        solution.calculate_fitness_score(self.data.scores)
        return solution 