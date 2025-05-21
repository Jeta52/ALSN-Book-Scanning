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
        
        # Get initial solution using GRASP
        print("Generating initial solution using GRASP...")
        current_solution = InitialSolution.generate_initial_solution_grasp(self.data)
        best_solution = current_solution.clone()
        
        self.stats['initial_score'] = current_solution.fitness_score
        print(f"Initial solution score: {current_solution.fitness_score}")
        
        # Track best operators
        best_destroy_op_count = {op: 0 for op in self.destroy_weights}
        best_repair_op_count = {op: 0 for op in self.repair_weights}
        
        while (time.time() - start_time) < time_limit:
            # Select destroy and repair operators based on weights
            destroy_op = self._select_operator(self.destroy_weights)
            repair_op = self._select_operator(self.repair_weights)
            
            # Create a copy of current solution
            temp_solution = current_solution.clone()
            
            # Apply destroy operator
            removed_libraries = destroy_op(temp_solution)
            
            # Apply repair operator
            new_solution = repair_op(temp_solution, removed_libraries)
            
            # Accept or reject the new solution
            if new_solution.fitness_score >= current_solution.fitness_score:
                current_solution = new_solution
                self.stats['improvements'] += 1
                
                # Update scores for successful operators
                if new_solution.fitness_score > best_solution.fitness_score:
                    # Global best improvement
                    self.destroy_scores[destroy_op] += self.score_weights['global_best']
                    self.repair_scores[repair_op] += self.score_weights['global_best']
                    
                    best_solution = new_solution.clone()
                    self.stats['time_to_best'] = time.time() - start_time
                    best_destroy_op_count[destroy_op] += 1
                    best_repair_op_count[repair_op] += 1
                    
                    # Reset iterations since improvement and reduce destroy percentage
                    self.iterations_since_improvement = 0
                    self.destroy_percentage = max(self.min_destroy_percentage, 
                                                self.destroy_percentage * 0.9)
                elif new_solution.fitness_score > current_solution.fitness_score:
                    improvement_percentage = ((new_solution.fitness_score - current_solution.fitness_score) 
                                           / current_solution.fitness_score)
                    if improvement_percentage > 0.01:  # More than 1% improvement
                        self.destroy_scores[destroy_op] += self.score_weights['significant']
                        self.repair_scores[repair_op] += self.score_weights['significant']
                    else:
                        self.destroy_scores[destroy_op] += self.score_weights['improvement']
                        self.repair_scores[repair_op] += self.score_weights['improvement']
                else:
                    # Equal to current solution
                    self.destroy_scores[destroy_op] += self.score_weights['accepted']
                    self.repair_scores[repair_op] += self.score_weights['accepted']
            # Implement simulated annealing acceptance criterion
            else:
                # Calculate probability of accepting worse solution
                temperature = max(0.01, 1.0 - (time.time() - start_time) / time_limit)  # Linear cooling
                delta = new_solution.fitness_score - current_solution.fitness_score
                probability = min(1.0, math.exp(delta / (temperature * current_solution.fitness_score * 0.01)))
                
                if random.random() < probability:
                    current_solution = new_solution
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
                print(f"\nProgress after {elapsed_time:.1f} seconds:")
                print(f"Iterations: {self.stats['iterations']}")
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
        print(f"Total iterations: {self.stats['iterations']}")
        print(f"Initial score: {self.stats['initial_score']}")
        print(f"Time to best solution: {self.stats['time_to_best']:.1f} seconds")
        print(f"Most successful destroy operator: {self.stats['best_destroy_op']}")
        print(f"Most successful repair operator: {self.stats['best_repair_op']}")
        
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
        num_to_remove = int(len(solution.signed_libraries) * self.destroy_percentage)
        if num_to_remove == 0:
            num_to_remove = 1
            
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
        curr_time = sum(self.data.libs[lib_id].signup_days for lib_id in solution.signed_libraries)
        
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
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
                solution.unsigned_libraries.remove(lib_id)
                solution.scanned_books_per_library[lib_id] = unique_books
                solution.scanned_books.update(unique_books)
                curr_time += library.signup_days
                
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def regret_score_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution using regret-based insertion"""
        curr_time = sum(self.data.libs[lib_id].signup_days for lib_id in solution.signed_libraries)
        
        while removed_libraries:
            max_regret = -1
            best_lib_id = None
            best_books = None
            
            for lib_id in removed_libraries:
                library = self.data.libs[lib_id]
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
                
            removed_libraries.remove(best_lib_id)
            solution.signed_libraries.append(best_lib_id)
            solution.unsigned_libraries.remove(best_lib_id)
            solution.scanned_books_per_library[best_lib_id] = best_books
            solution.scanned_books.update(best_books)
            curr_time += self.data.libs[best_lib_id].signup_days
            
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def max_books_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by maximizing the number of books that can be scanned"""
        curr_time = sum(self.data.libs[lib_id].signup_days for lib_id in solution.signed_libraries)
        
        library_potentials = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
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
            solution.signed_libraries.append(lib_id)
            solution.unsigned_libraries.remove(lib_id)
            solution.scanned_books_per_library[lib_id] = books
            solution.scanned_books.update(books)
            curr_time += self.data.libs[lib_id].signup_days
            
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def random_feasible_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Repair solution by randomly inserting libraries while maintaining feasibility"""
        curr_time = sum(self.data.libs[lib_id].signup_days for lib_id in solution.signed_libraries)
        
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