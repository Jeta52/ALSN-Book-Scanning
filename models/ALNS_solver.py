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
        self.max_destroy_percentage = 0.5
        self.iterations_since_improvement = 0
        
        # Reaction factor for weight adjustments (how quickly weights adapt)
        self.reaction_factor = 0.1
        
        # Segment size for weight adjustments
        self.segment_size = 100
        
        # Score weights for different improvement types
        self.score_weights = {
            'global_best': 15,    # New best solution found
            'significant': 8,     # Improvement > 1% over current
            'improvement': 5,     # Any improvement over current
            'accepted': 2,        # Accepted but not better
            'rejected': 0         # Not accepted
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
            'best_repair_op': None
        }
        
    def solve(self, time_limit=600) -> Solution:  # 10 minutes = 600 seconds
        """
        Main ALNS solving method that runs for 10 minutes per instance
        """
        start_time = time.time()
        last_print_time = start_time
        print_interval = 30  # Print progress every 30 seconds
        
        # Allocate time: 10% for initial solution, 90% for ALNS
        initial_solution_time = min(60, time_limit * 0.1)  # Max 60 seconds for initial solution
        
        # Get initial solution using GRASP with limited time
        print("Generating initial solution using GRASP...")
        print(f"Time allocated for initial solution: {initial_solution_time:.1f} seconds")
        current_solution = InitialSolution.generate_initial_solution_grasp(self.data, max_time=initial_solution_time)
        
        # Ensure fitness score is calculated
        current_solution.calculate_fitness_score(self.data.scores)
        best_solution = current_solution.clone()
        best_solution.calculate_fitness_score(self.data.scores)
        
        self.stats['initial_score'] = current_solution.fitness_score
        self.stats['final_score'] = best_solution.fitness_score  # Initialize final score
        print(f"Initial solution score: {current_solution.fitness_score}")
        print(f"Initial solution has {len(current_solution.signed_libraries)} signed libraries")
        print(f"Initial solution has {len(current_solution.scanned_books)} scanned books")
        
        # Calculate remaining time for ALNS
        time_after_initial = time.time() - start_time
        remaining_time = time_limit - time_after_initial
        print(f"Time remaining for ALNS: {remaining_time:.1f} seconds")
        
        if remaining_time <= 0:
            print("Warning: No time remaining for ALNS optimization!")
            return best_solution
        
        # Track best operators
        best_destroy_op_count = {op: 0 for op in self.destroy_weights}
        best_repair_op_count = {op: 0 for op in self.repair_weights}
        
        iterations_count = 0
        accepted_count = 0
        improvement_count = 0
        
        print(f"Starting ALNS loop with time limit: {remaining_time:.1f} seconds")
        
        alns_start_time = time.time()
        while (time.time() - alns_start_time) < remaining_time:
            iterations_count += 1
            
            if iterations_count == 1:
                print("Entered main ALNS loop successfully")
            
            # Select destroy and repair operators based on weights
            destroy_op = self._select_operator(self.destroy_weights)
            repair_op = self._select_operator(self.repair_weights)
            
            if iterations_count <= 5:  # Debug first few iterations
                print(f"Iteration {iterations_count}: Selected destroy={destroy_op.__name__}, repair={repair_op.__name__}")
            
            # Create a copy of current solution
            temp_solution = current_solution.clone()
            
            # Apply destroy operator
            removed_libraries = destroy_op(temp_solution)
            
            if iterations_count <= 5:
                print(f"Iteration {iterations_count}: Destroyed {len(removed_libraries)} libraries")
            
            # Apply repair operator
            new_solution = repair_op(temp_solution, removed_libraries)
            
            # Ensure the new solution has a proper fitness score
            if hasattr(new_solution, 'fitness_score') and new_solution.fitness_score is not None:
                # Fitness score already calculated in repair operator
                pass
            else:
                new_solution.calculate_fitness_score(self.data.scores)
            
            if iterations_count <= 5:
                print(f"Iteration {iterations_count}: Repaired solution, new score: {new_solution.fitness_score}")
            
            # Debug: Print some information every 100 iterations
            if iterations_count % 100 == 0:
                print(f"Iteration {iterations_count}: Current={current_solution.fitness_score}, New={new_solution.fitness_score}, Best={best_solution.fitness_score}")
            
            # Accept or reject the new solution
            if new_solution.fitness_score > current_solution.fitness_score:
                # Clear improvement
                current_solution = new_solution
                self.stats['improvements'] += 1
                improvement_count += 1
                accepted_count += 1
                
                # Update scores for successful operators
                if new_solution.fitness_score > best_solution.fitness_score:
                    # Global best improvement
                    self.destroy_scores[destroy_op] += self.score_weights['global_best']
                    self.repair_scores[repair_op] += self.score_weights['global_best']
                    
                    best_solution = new_solution.clone()
                    self.stats['time_to_best'] = time.time() - start_time
                    best_destroy_op_count[destroy_op] += 1
                    best_repair_op_count[repair_op] += 1
                    
                    print(f"*** NEW BEST SOLUTION FOUND at iteration {iterations_count}: {best_solution.fitness_score} ***")
                    
                    # Reset iterations since improvement and reduce destroy percentage
                    self.iterations_since_improvement = 0
                    self.destroy_percentage = max(self.min_destroy_percentage, 
                                                self.destroy_percentage * 0.9)
                else:
                    improvement_percentage = ((new_solution.fitness_score - current_solution.fitness_score) 
                                           / current_solution.fitness_score)
                    if improvement_percentage > 0.01:  # More than 1% improvement
                        self.destroy_scores[destroy_op] += self.score_weights['significant']
                        self.repair_scores[repair_op] += self.score_weights['significant']
                    else:
                        self.destroy_scores[destroy_op] += self.score_weights['improvement']
                        self.repair_scores[repair_op] += self.score_weights['improvement']
                        
            elif new_solution.fitness_score == current_solution.fitness_score:
                # Equal to current solution
                current_solution = new_solution
                accepted_count += 1
                self.destroy_scores[destroy_op] += self.score_weights['accepted']
                self.repair_scores[repair_op] += self.score_weights['accepted']
                
            # Implement simulated annealing acceptance criterion for worse solutions
            else:
                # Calculate probability of accepting worse solution
                temperature = max(0.01, 1.0 - (time.time() - start_time) / time_limit)  # Linear cooling
                delta = new_solution.fitness_score - current_solution.fitness_score
                probability = min(1.0, math.exp(delta / (temperature * current_solution.fitness_score * 0.01)))
                
                if random.random() < probability:
                    current_solution = new_solution
                    accepted_count += 1
                    # Accepted worse solution
                    self.destroy_scores[destroy_op] += self.score_weights['accepted']
                    self.repair_scores[repair_op] += self.score_weights['accepted']
                else:
                    # Rejected solution
                    self.destroy_scores[destroy_op] += self.score_weights['rejected']
                    self.repair_scores[repair_op] += self.score_weights['rejected']
            
            # Increase destroy percentage if stuck in local optima
            self.iterations_since_improvement += 1
            if self.iterations_since_improvement > 50:
                self.destroy_percentage = min(self.max_destroy_percentage, 
                                            self.destroy_percentage * 1.1)
                self.iterations_since_improvement = 0
            
            # Update weights periodically
            if self.stats['iterations'] % self.segment_size == 0:
                self._update_weights()
            
            self.stats['iterations'] += 1
            
            # Print progress every 30 seconds
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                elapsed_time = current_time - start_time
                acceptance_rate = (accepted_count / iterations_count) * 100 if iterations_count > 0 else 0
                improvement_rate = (improvement_count / iterations_count) * 100 if iterations_count > 0 else 0
                
                print(f"\nProgress after {elapsed_time:.1f} seconds:")
                print(f"Iterations: {iterations_count}")
                print(f"Acceptance rate: {acceptance_rate:.1f}%")
                print(f"Improvement rate: {improvement_rate:.1f}%")
                print(f"Current score: {current_solution.fitness_score}")
                print(f"Best score: {best_solution.fitness_score}")
                print(f"Destroy percentage: {self.destroy_percentage:.3f}")
                print(f"Time since last improvement: {current_time - (start_time + self.stats['time_to_best']):.1f}s")
                
                # Print top destroy operators
                top_destroy = sorted(self.destroy_weights.items(), key=lambda x: x[1], reverse=True)[:2]
                top_repair = sorted(self.repair_weights.items(), key=lambda x: x[1], reverse=True)[:2]
                print(f"Top destroy operators: {top_destroy[0][0].__name__} ({top_destroy[0][1]:.2f}), {top_destroy[1][0].__name__} ({top_destroy[1][1]:.2f})")
                print(f"Top repair operators: {top_repair[0][0].__name__} ({top_repair[0][1]:.2f}), {top_repair[1][0].__name__} ({top_repair[1][1]:.2f})")
                
                last_print_time = current_time
        
        # Final statistics
        self.stats['final_score'] = best_solution.fitness_score
        
        # Determine best operators
        self.stats['best_destroy_op'] = max(best_destroy_op_count.items(), key=lambda x: x[1])[0].__name__
        self.stats['best_repair_op'] = max(best_repair_op_count.items(), key=lambda x: x[1])[0].__name__
        
        total_time = time.time() - start_time
        print("\nFinal Statistics:")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Total iterations: {iterations_count}")
        print(f"Total accepted solutions: {accepted_count}")
        print(f"Total improvements: {improvement_count}")
        
        if iterations_count > 0:
            print(f"Acceptance rate: {(accepted_count/iterations_count)*100:.1f}%")
            print(f"Improvement rate: {(improvement_count/iterations_count)*100:.1f}%")
        else:
            print("Acceptance rate: N/A (no iterations completed)")
            print("Improvement rate: N/A (no iterations completed)")
            
        print(f"Initial score: {self.stats['initial_score']}")
        print(f"Final best score: {self.stats['final_score']}")
        print(f"Improvement: {self.stats['final_score'] - self.stats['initial_score']}")
        print(f"Time to best solution: {self.stats['time_to_best']:.1f} seconds")
        
        if iterations_count > 0:
            print(f"Most successful destroy operator: {self.stats['best_destroy_op']}")
            print(f"Most successful repair operator: {self.stats['best_repair_op']}")
        else:
            print("No operators were executed")
        
        return best_solution
    
    def _select_operator(self, weights: Dict) -> callable:
        """Select an operator based on weights and usage history"""
        operators = list(weights.keys())
        weights_list = []
        
        # Get usage counts
        usage_dict = self.destroy_usage if operators[0] in self.destroy_weights else self.repair_usage
        total_usage = sum(usage_dict.values()) + len(operators)  # Add len(operators) to avoid division by zero
        
        for op in operators:
            # Base weight from adaptive scoring
            weight = weights[op]
            
            # Adjust weight based on usage frequency to promote diversity
            usage = usage_dict[op]
            usage_penalty = (usage / total_usage) ** 2  # Quadratic penalty for frequently used operators
            
            # Final weight combines base weight with usage penalty
            final_weight = weight * (1.0 - usage_penalty)
            weights_list.append(max(0.1, final_weight))  # Ensure minimum weight of 0.1
            
        # Update usage counter for selected operator
        selected = random.choices(operators, weights=weights_list, k=1)[0]
        if selected in self.destroy_weights:
            self.destroy_usage[selected] += 1
        else:
            self.repair_usage[selected] += 1
            
        return selected
    
    def _update_weights(self):
        """Update weights of operators based on their success using reaction factor"""
        # Only update weights every segment_size iterations
        if self.stats['iterations'] % self.segment_size != 0:
            return
            
        # Update destroy weights
        total_destroy_score = sum(self.destroy_scores.values()) + 1e-10
        for op in self.destroy_weights:
            if self.destroy_usage[op] > 0:  # Only update if operator was used
                normalized_score = self.destroy_scores[op] / self.destroy_usage[op]
                normalized_score = normalized_score / total_destroy_score
                
                # Update weight using reaction factor
                self.destroy_weights[op] = (
                    (1 - self.reaction_factor) * self.destroy_weights[op] +
                    self.reaction_factor * normalized_score
                )
            
        # Update repair weights similarly
        total_repair_score = sum(self.repair_scores.values()) + 1e-10
        for op in self.repair_weights:
            if self.repair_usage[op] > 0:  # Only update if operator was used
                normalized_score = self.repair_scores[op] / self.repair_usage[op]
                normalized_score = normalized_score / total_repair_score
                
                # Update weight using reaction factor
                self.repair_weights[op] = (
                    (1 - self.reaction_factor) * self.repair_weights[op] +
                    self.reaction_factor * normalized_score
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
            
        num_to_remove = int(len(solution.signed_libraries) * self.destroy_percentage)
        if num_to_remove == 0:
            num_to_remove = 1
            
        num_to_remove = min(num_to_remove, len(solution.signed_libraries))
        indices_to_remove = random.sample(range(len(solution.signed_libraries)), num_to_remove)
        removed_libraries = []
        
        for idx in sorted(indices_to_remove, reverse=True):
            lib_id = solution.signed_libraries.pop(idx)
            removed_libraries.append(lib_id)
            if lib_id not in solution.unsigned_libraries:
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
            efficiency = score / (library.signup_days + 1e-10)
            library_efficiencies.append((lib_id, efficiency))
            
        library_efficiencies.sort(key=lambda x: x[1])
        num_to_remove = int(len(solution.signed_libraries) * self.destroy_percentage)
        if num_to_remove == 0:
            num_to_remove = 1
            
        num_to_remove = min(num_to_remove, len(solution.signed_libraries))
        removed_libraries = []
        for i in range(num_to_remove):
            lib_id = library_efficiencies[i][0]
            solution.signed_libraries.remove(lib_id)
            if lib_id not in solution.unsigned_libraries:
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
                if sum(1 for other_lib_id, other_books in solution.scanned_books_per_library.items()
                      if other_lib_id != lib_id and book_id in other_books) > 0:
                    duplicates += 1
            library_duplicates.append((lib_id, duplicates))
            
        library_duplicates.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = int(len(solution.signed_libraries) * self.destroy_percentage)
        if num_to_remove == 0:
            num_to_remove = 1
            
        num_to_remove = min(num_to_remove, len(solution.signed_libraries))
        removed_libraries = []
        for i in range(num_to_remove):
            lib_id = library_duplicates[i][0]
            solution.signed_libraries.remove(lib_id)
            if lib_id not in solution.unsigned_libraries:
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
        num_to_remove = int(len(solution.signed_libraries) * self.destroy_percentage)
        if num_to_remove == 0:
            num_to_remove = 1
            
        num_to_remove = min(num_to_remove, len(solution.signed_libraries))
        removed_libraries = []
        for i in range(num_to_remove):
            lib_id = library_overlaps[i][0]
            solution.signed_libraries.remove(lib_id)
            if lib_id not in solution.unsigned_libraries:
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
        # Sort removed libraries by potential score
        library_candidates = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            unique_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            if unique_books:
                potential_score = sum(self.data.scores[b] for b in unique_books)
                library_candidates.append((potential_score, lib_id))
        
        library_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Try to add libraries in order of potential score
        for _, lib_id in library_candidates:
            library = self.data.libs[lib_id]
            
            # Calculate current time based on current library order
            curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
            
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            # Get unique books and their scores
            unique_books = sorted(
                [b.id for b in library.books if b.id not in solution.scanned_books],
                key=lambda b: self.data.scores[b],
                reverse=True
            )[:max_books]
            
            if unique_books:
                solution.signed_libraries.append(lib_id)
                if lib_id in solution.unsigned_libraries:
                    solution.unsigned_libraries.remove(lib_id)
                solution.scanned_books_per_library[lib_id] = unique_books
                solution.scanned_books.update(unique_books)
                
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def regret_score_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution using regret-based insertion"""
        remaining_libraries = removed_libraries.copy()
        
        while remaining_libraries:
            max_regret = -1
            best_lib_id = None
            best_books = None
            
            for lib_id in remaining_libraries:
                library = self.data.libs[lib_id]
                
                # Calculate current time based on current library order
                curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
                
                if curr_time + library.signup_days >= self.data.num_days:
                    continue
                    
                time_left = self.data.num_days - (curr_time + library.signup_days)
                max_books = time_left * library.books_per_day
                
                # Calculate regret (difference between best and second best position)
                available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
                if not available_books:
                    continue
                    
                sorted_books = sorted(available_books, key=lambda b: self.data.scores[b], reverse=True)[:max_books]
                best_score = sum(self.data.scores[b] for b in sorted_books)
                
                # Calculate regret as the score that would be lost if not chosen now
                regret = best_score
                
                if regret > max_regret:
                    max_regret = regret
                    best_lib_id = lib_id
                    best_books = sorted_books
            
            if best_lib_id is None:
                break
                
            remaining_libraries.remove(best_lib_id)
            solution.signed_libraries.append(best_lib_id)
            if best_lib_id in solution.unsigned_libraries:
                solution.unsigned_libraries.remove(best_lib_id)
            solution.scanned_books_per_library[best_lib_id] = best_books
            solution.scanned_books.update(best_books)
            
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def max_books_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by maximizing the number of books that can be scanned"""
        library_potentials = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            
            # Calculate current time based on current library order
            curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
            
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            potential_books = len(available_books[:max_books])
            
            if potential_books > 0:
                library_potentials.append((lib_id, potential_books, available_books[:max_books]))
                
        # Sort by number of potential books
        library_potentials.sort(key=lambda x: x[1], reverse=True)
        
        for lib_id, _, books in library_potentials:
            # Recalculate current time before adding each library
            curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
            library = self.data.libs[lib_id]
            
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            # Recalculate available books based on current scanned books
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            books_to_add = available_books[:max_books]
            
            if books_to_add:
                solution.signed_libraries.append(lib_id)
                if lib_id in solution.unsigned_libraries:
                    solution.unsigned_libraries.remove(lib_id)
                solution.scanned_books_per_library[lib_id] = books_to_add
                solution.scanned_books.update(books_to_add)
            
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def random_feasible_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by randomly inserting libraries while maintaining feasibility"""
        random.shuffle(removed_libraries)
        
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            
            # Calculate current time based on current library order
            curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
            
            if curr_time + library.signup_days >= self.data.num_days:
                continue
                
            time_left = self.data.num_days - (curr_time + library.signup_days)
            max_books = time_left * library.books_per_day
            
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            books_to_scan = available_books[:max_books]
            
            if books_to_scan:
                solution.signed_libraries.append(lib_id)
                if lib_id in solution.unsigned_libraries:
                    solution.unsigned_libraries.remove(lib_id)
                solution.scanned_books_per_library[lib_id] = books_to_scan
                solution.scanned_books.update(books_to_scan)
                
        solution.calculate_fitness_score(self.data.scores)
        return solution 