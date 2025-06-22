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
        self.min_destroy_percentage = 0.05
        self.max_destroy_percentage = 0.6
        self.iterations_since_improvement = 0
        
        self.reaction_factor = 0.15
        
        # Segment size for weight adjustments
        self.segment_size = 50
        
        self.score_weights = {
            'global_best': 20,    
            'significant': 10,    
            'improvement': 6,   
            'accepted': 3,       
            'rejected': 0        
        }
        
        self.destroy_weights = {
            self.random_library_removal: 1.0,
            self.low_efficiency_library_removal: 1.0,
            self.duplicate_book_removal: 1.0,
            self.overlap_based_removal: 1.0,
            # New destroy operators
            self.time_based_removal: 1.0,
            self.cluster_based_removal: 1.0,
            self.late_starting_removal: 1.0,
            self.low_score_density_removal: 1.0,
            self.adaptive_worst_removal: 1.0,
            self.bottleneck_removal: 1.0
        }
        
        # Weights for repair operators (expanded with new operators)
        self.repair_weights = {
            self.greedy_max_unique_score_repair: 1.0,
            self.regret_score_repair: 1.0,
            self.max_books_repair: 1.0,
            self.random_feasible_repair: 1.0,
            # New repair operators
            self.best_fit_repair: 1.0,
            self.score_density_repair: 1.0,
            self.time_efficient_repair: 1.0,
            self.balanced_repair: 1.0,
            self.local_search_repair: 1.0,
            self.opportunity_cost_repair: 1.0
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
        
    def solve(self, time_limit=600) -> Solution:
        """
        Main ALNS solving method that runs for 10 minutes per instance
        """
        start_time = time.time()
        last_print_time = start_time
        print_interval = 30
        initial_solution_time = 60 
        
        # Get initial solution using GRASP with limited time
        print("Generating initial solution using GRASP...")
        print(f"Time allocated for initial solution: {initial_solution_time:.1f} seconds")
        current_solution = InitialSolution.generate_initial_solution_grasp(self.data, max_time=initial_solution_time)
        current_solution.calculate_fitness_score(self.data.scores)
        best_solution = current_solution.clone()
        best_solution.calculate_fitness_score(self.data.scores)
        
        self.stats['initial_score'] = current_solution.fitness_score
        self.stats['final_score'] = best_solution.fitness_score
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
            
            if iterations_count <= 5:
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
        """Enhanced weight update with better normalization and diversity promotion"""
        # Only update weights every segment_size iterations
        if self.stats['iterations'] % self.segment_size != 0:
            return
            
        max_destroy_score = max(self.destroy_scores.values()) if self.destroy_scores.values() else 1
        
        for op in self.destroy_weights:
            if self.destroy_usage[op] > 0:
                avg_score = self.destroy_scores[op] / self.destroy_usage[op]
                normalized_score = avg_score / (max_destroy_score + 1e-10)
                diversity_bonus = 1.0 + (0.5 * (1.0 - self.destroy_usage[op] / sum(self.destroy_usage.values())))
                
                # Update weight with enhanced reaction factor
                new_weight = normalized_score * diversity_bonus
                self.destroy_weights[op] = (
                    (1 - self.reaction_factor) * self.destroy_weights[op] +
                    self.reaction_factor * new_weight
                )
                
                # Ensure minimum weight for exploration
                self.destroy_weights[op] = max(0.1, self.destroy_weights[op])
            
        # Enhanced repair weight updates (similar approach)
        max_repair_score = max(self.repair_scores.values()) if self.repair_scores.values() else 1
        
        for op in self.repair_weights:
            if self.repair_usage[op] > 0:
                avg_score = self.repair_scores[op] / self.repair_usage[op]
                normalized_score = avg_score / (max_repair_score + 1e-10)
                
                diversity_bonus = 1.0 + (0.5 * (1.0 - self.repair_usage[op] / sum(self.repair_usage.values())))
                
                new_weight = normalized_score * diversity_bonus
                self.repair_weights[op] = (
                    (1 - self.reaction_factor) * self.repair_weights[op] +
                    self.reaction_factor * new_weight
                )
                
                self.repair_weights[op] = max(0.1, self.repair_weights[op])
            
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
        """Enhanced version with better efficiency calculation"""
        if not solution.signed_libraries:
            return []
            
        library_efficiencies = []
        for lib_id in solution.signed_libraries:
            library = self.data.libs[lib_id]
            books = solution.scanned_books_per_library.get(lib_id, [])
            
            if not books:
                efficiency = 0
            else:
                score = sum(self.data.scores[book_id] for book_id in books)
                # Consider both time cost and book count
                time_cost = library.signup_days + len(books) / library.books_per_day
                efficiency = score / (time_cost + 1e-10)
                
                # Penalty for duplicate books
                unique_books = len([b for b in books if b not in solution.scanned_books or 
                                  sum(1 for other_lib, other_books in solution.scanned_books_per_library.items() 
                                      if other_lib != lib_id and b in other_books) == 0])
                duplicate_penalty = unique_books / len(books) if books else 1
                efficiency *= duplicate_penalty
                
            library_efficiencies.append((lib_id, efficiency))
            
        library_efficiencies.sort(key=lambda x: x[1])
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
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
    
    # New Destroy Operators
    def time_based_removal(self, solution: Solution) -> List[int]:
        """Remove libraries that consume disproportionate time relative to their score contribution"""
        if not solution.signed_libraries:
            return []
            
        library_time_efficiency = []
        for lib_id in solution.signed_libraries:
            library = self.data.libs[lib_id]
            books = solution.scanned_books_per_library.get(lib_id, [])
            
            if not books:
                time_efficiency = 0
            else:
                score = sum(self.data.scores[book_id] for book_id in books)
                total_time = library.signup_days + len(books) / library.books_per_day
                time_efficiency = score / total_time
                
            library_time_efficiency.append((lib_id, time_efficiency))
            
        library_time_efficiency.sort(key=lambda x: x[1])
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        
        return self._remove_libraries(solution, [lib for lib, _ in library_time_efficiency[:num_to_remove]])
    
    def cluster_based_removal(self, solution: Solution) -> List[int]:
        """Remove clusters of libraries with high book overlap"""
        if len(solution.signed_libraries) < 2:
            return []
            
        # Build similarity matrix based on book overlap
        similarities = {}
        for i, lib1 in enumerate(solution.signed_libraries):
            for j, lib2 in enumerate(solution.signed_libraries[i+1:], i+1):
                books1 = set(solution.scanned_books_per_library.get(lib1, []))
                books2 = set(solution.scanned_books_per_library.get(lib2, []))
                
                if books1 or books2:
                    overlap = len(books1.intersection(books2))
                    total_books = len(books1.union(books2))
                    similarity = overlap / total_books if total_books > 0 else 0
                    similarities[(lib1, lib2)] = similarity
        
        # Find the cluster with highest internal similarity
        if not similarities:
            return self.random_library_removal(solution)
            
        # Simple clustering: find the pair with highest similarity and expand
        best_pair = max(similarities.items(), key=lambda x: x[1])
        cluster = list(best_pair[0])
        
        # Expand cluster by adding libraries with high similarity to cluster members
        for lib_id in solution.signed_libraries:
            if lib_id not in cluster:
                avg_similarity = 0
                count = 0
                for cluster_lib in cluster:
                    key = (min(lib_id, cluster_lib), max(lib_id, cluster_lib))
                    if key in similarities:
                        avg_similarity += similarities[key]
                        count += 1
                
                if count > 0 and avg_similarity / count > 0.3:  # Threshold for cluster membership
                    cluster.append(lib_id)
        
        # Remove a portion of the cluster
        num_to_remove = min(len(cluster), max(1, int(len(solution.signed_libraries) * self.destroy_percentage)))
        libraries_to_remove = random.sample(cluster, num_to_remove)
        
        return self._remove_libraries(solution, libraries_to_remove)
    
    def late_starting_removal(self, solution: Solution) -> List[int]:
        """Remove libraries that start late in the scanning timeline"""
        if not solution.signed_libraries:
            return []
            
        # Calculate start times for each library
        library_start_times = []
        current_time = 0
        
        for lib_id in solution.signed_libraries:
            library_start_times.append((lib_id, current_time))
            current_time += self.data.libs[lib_id].signup_days
            
        # Sort by start time (descending) and remove latest starting libraries
        library_start_times.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        
        libraries_to_remove = [lib for lib, _ in library_start_times[:num_to_remove]]
        return self._remove_libraries(solution, libraries_to_remove)
    
    def low_score_density_removal(self, solution: Solution) -> List[int]:
        """Remove libraries with low score per book ratio"""
        if not solution.signed_libraries:
            return []
            
        library_densities = []
        for lib_id in solution.signed_libraries:
            books = solution.scanned_books_per_library.get(lib_id, [])
            if books:
                score = sum(self.data.scores[book_id] for book_id in books)
                density = score / len(books)
            else:
                density = 0
            library_densities.append((lib_id, density))
            
        library_densities.sort(key=lambda x: x[1])
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        
        libraries_to_remove = [lib for lib, _ in library_densities[:num_to_remove]]
        return self._remove_libraries(solution, libraries_to_remove)
    
    def adaptive_worst_removal(self, solution: Solution) -> List[int]:
        """Adaptively remove worst libraries based on current solution characteristics"""
        if not solution.signed_libraries:
            return []
            
        # Calculate multiple metrics for each library
        library_metrics = []
        for lib_id in solution.signed_libraries:
            library = self.data.libs[lib_id]
            books = solution.scanned_books_per_library.get(lib_id, [])
            
            if books:
                score = sum(self.data.scores[book_id] for book_id in books)
                time_cost = library.signup_days + len(books) / library.books_per_day
                efficiency = score / time_cost
                density = score / len(books)
                
                # Count unique books (not in other libraries)
                unique_count = sum(1 for book_id in books if 
                                 sum(1 for other_lib, other_books in solution.scanned_books_per_library.items()
                                     if other_lib != lib_id and book_id in other_books) == 0)
                uniqueness = unique_count / len(books) if books else 0
                
                # Composite score (lower is worse)
                composite_score = efficiency * 0.4 + density * 0.3 + uniqueness * 0.3
            else:
                composite_score = 0
                
            library_metrics.append((lib_id, composite_score))
            
        library_metrics.sort(key=lambda x: x[1])
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        
        libraries_to_remove = [lib for lib, _ in library_metrics[:num_to_remove]]
        return self._remove_libraries(solution, libraries_to_remove)
    
    def bottleneck_removal(self, solution: Solution) -> List[int]:
        """Remove libraries that create bottlenecks in the scanning timeline"""
        if not solution.signed_libraries:
            return []
            
        library_bottlenecks = []
        for lib_id in solution.signed_libraries:
            library = self.data.libs[lib_id]
            books = solution.scanned_books_per_library.get(lib_id, [])
            
            # Calculate bottleneck score based on signup time vs scanning time ratio
            signup_time = library.signup_days
            scanning_time = len(books) / library.books_per_day if books and library.books_per_day > 0 else 0
            
            # High signup time relative to scanning time indicates a bottleneck
            bottleneck_score = signup_time / (scanning_time + 1) if scanning_time > 0 else signup_time
            library_bottlenecks.append((lib_id, bottleneck_score))
            
        library_bottlenecks.sort(key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, int(len(solution.signed_libraries) * self.destroy_percentage))
        
        libraries_to_remove = [lib for lib, _ in library_bottlenecks[:num_to_remove]]
        return self._remove_libraries(solution, libraries_to_remove)

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
    
    # New Repair Operators
    def best_fit_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Insert libraries at positions that minimize timeline disruption"""
        # Sort libraries by their potential contribution
        library_candidates = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            if available_books:
                potential_score = sum(self.data.scores[b] for b in available_books)
                efficiency = potential_score / (library.signup_days + len(available_books) / library.books_per_day)
                library_candidates.append((lib_id, efficiency, potential_score))
        
        library_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for lib_id, _, _ in library_candidates:
            best_position = self._find_best_insertion_position(solution, lib_id)
            if best_position is not None:
                self._insert_library_at_position(solution, lib_id, best_position)
        
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def score_density_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Prioritize libraries with highest score per book ratio"""
        library_densities = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            if available_books:
                score = sum(self.data.scores[b] for b in available_books)
                density = score / len(available_books)
                library_densities.append((lib_id, density, available_books))
        
        library_densities.sort(key=lambda x: x[1], reverse=True)
        
        for lib_id, _, available_books in library_densities:
            if self._can_add_library(solution, lib_id):
                self._add_library_to_solution(solution, lib_id, available_books)
        
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def time_efficient_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Prioritize libraries that can be completed quickly"""
        library_times = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            if available_books:
                total_time = library.signup_days + len(available_books) / library.books_per_day
                score = sum(self.data.scores[b] for b in available_books)
                library_times.append((lib_id, total_time, score, available_books))
        
        # Sort by time (ascending) but weight by score
        library_times.sort(key=lambda x: x[1] / (x[2] + 1))
        
        for lib_id, _, _, available_books in library_times:
            if self._can_add_library(solution, lib_id):
                self._add_library_to_solution(solution, lib_id, available_books)
        
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def balanced_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Balance between score, time efficiency, and book count"""
        library_scores = []
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            if available_books:
                score = sum(self.data.scores[b] for b in available_books)
                time_cost = library.signup_days + len(available_books) / library.books_per_day
                efficiency = score / time_cost
                book_count = len(available_books)
                
                # Balanced score: 50% efficiency, 30% raw score, 20% book count
                balanced_score = 0.5 * efficiency + 0.3 * score + 0.2 * book_count
                library_scores.append((lib_id, balanced_score, available_books))
        
        library_scores.sort(key=lambda x: x[1], reverse=True)
        
        for lib_id, _, available_books in library_scores:
            if self._can_add_library(solution, lib_id):
                self._add_library_to_solution(solution, lib_id, available_books)
        
        solution.calculate_fitness_score(self.data.scores)
        return solution
    
    def local_search_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Apply local search optimization after greedy insertion"""
        # First, do greedy insertion
        solution = self.greedy_max_unique_score_repair(solution, removed_libraries)
        
        # Then apply local search improvements
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try swapping adjacent libraries
            for i in range(len(solution.signed_libraries) - 1):
                original_score = solution.fitness_score
                
                # Swap libraries i and i+1
                temp_solution = solution.clone()
                temp_solution.signed_libraries[i], temp_solution.signed_libraries[i+1] = \
                    temp_solution.signed_libraries[i+1], temp_solution.signed_libraries[i]
                
                # Recalculate scanned books for the swapped solution
                self._recalculate_scanned_books(temp_solution)
                temp_solution.calculate_fitness_score(self.data.scores)
                
                if temp_solution.fitness_score > original_score:
                    solution = temp_solution
                    improved = True
                    break
        
        return solution
    
    def opportunity_cost_repair(self, solution: Solution, removed_libraries: List[int]) -> Solution:
        """Consider opportunity cost of not selecting libraries"""
        library_opportunities = []
        
        for lib_id in removed_libraries:
            library = self.data.libs[lib_id]
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            
            if available_books:
                # Calculate direct benefit
                direct_score = sum(self.data.scores[b] for b in available_books)
                
                # Calculate opportunity cost (what we lose by not taking it)
                opportunity_cost = 0
                for other_lib_id in removed_libraries:
                    if other_lib_id != lib_id:
                        other_library = self.data.libs[other_lib_id]
                        overlapping_books = [b.id for b in other_library.books 
                                           if b.id in available_books and b.id not in solution.scanned_books]
                        if overlapping_books:
                            opportunity_cost += sum(self.data.scores[b] for b in overlapping_books)
                
                total_value = direct_score + opportunity_cost * 0.3  # Weight opportunity cost
                library_opportunities.append((lib_id, total_value, available_books))
        
        library_opportunities.sort(key=lambda x: x[1], reverse=True)
        
        for lib_id, _, available_books in library_opportunities:
            if self._can_add_library(solution, lib_id):
                self._add_library_to_solution(solution, lib_id, available_books)
        
        solution.calculate_fitness_score(self.data.scores)
        return solution

    # Helper methods for new operators
    def _remove_libraries(self, solution: Solution, libraries_to_remove: List[int]) -> List[int]:
        """Helper method to remove libraries from solution"""
        removed_libraries = []
        for lib_id in libraries_to_remove:
            if lib_id in solution.signed_libraries:
                solution.signed_libraries.remove(lib_id)
                if lib_id not in solution.unsigned_libraries:
                    solution.unsigned_libraries.append(lib_id)
                removed_libraries.append(lib_id)
                
                if lib_id in solution.scanned_books_per_library:
                    books = solution.scanned_books_per_library[lib_id]
                    solution.scanned_books.difference_update(books)
                    del solution.scanned_books_per_library[lib_id]
        
        return removed_libraries
    
    def _find_best_insertion_position(self, solution: Solution, lib_id: int) -> int:
        """Find the best position to insert a library in the timeline"""
        best_position = None
        best_score_improvement = -1
        
        library = self.data.libs[lib_id]
        
        # Try inserting at each position
        for pos in range(len(solution.signed_libraries) + 1):
            temp_solution = solution.clone()
            temp_solution.signed_libraries.insert(pos, lib_id)
            
            # Calculate if this position is feasible and beneficial
            curr_time = 0
            feasible = True
            
            for i, signed_lib_id in enumerate(temp_solution.signed_libraries):
                lib = self.data.libs[signed_lib_id]
                curr_time += lib.signup_days
                
                if curr_time >= self.data.num_days:
                    feasible = False
                    break
                    
                if signed_lib_id == lib_id:
                    # Calculate potential books for the inserted library
                    time_left = self.data.num_days - curr_time
                    max_books = time_left * lib.books_per_day
                    available_books = [b.id for b in lib.books if b.id not in solution.scanned_books]
                    books_to_scan = available_books[:max_books]
                    
                    if books_to_scan:
                        score_improvement = sum(self.data.scores[b] for b in books_to_scan)
                        if score_improvement > best_score_improvement:
                            best_score_improvement = score_improvement
                            best_position = pos
            
            if not feasible:
                continue
        
        return best_position
    
    def _insert_library_at_position(self, solution: Solution, lib_id: int, position: int):
        """Insert library at specific position and update scanned books"""
        solution.signed_libraries.insert(position, lib_id)
        if lib_id in solution.unsigned_libraries:
            solution.unsigned_libraries.remove(lib_id)
        
        # Recalculate all scanned books due to position change
        self._recalculate_scanned_books(solution)
    
    def _can_add_library(self, solution: Solution, lib_id: int) -> bool:
        """Check if library can be added to solution"""
        curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
        library = self.data.libs[lib_id]
        return curr_time + library.signup_days < self.data.num_days
    
    def _add_library_to_solution(self, solution: Solution, lib_id: int, available_books: List[int]):
        """Add library to solution with specified books"""
        library = self.data.libs[lib_id]
        curr_time = sum(self.data.libs[signed_lib_id].signup_days for signed_lib_id in solution.signed_libraries)
        
        if curr_time + library.signup_days >= self.data.num_days:
            return
            
        time_left = self.data.num_days - (curr_time + library.signup_days)
        max_books = time_left * library.books_per_day
        books_to_scan = sorted(available_books, key=lambda b: self.data.scores[b], reverse=True)[:max_books]
        
        if books_to_scan:
            solution.signed_libraries.append(lib_id)
            if lib_id in solution.unsigned_libraries:
                solution.unsigned_libraries.remove(lib_id)
            solution.scanned_books_per_library[lib_id] = books_to_scan
            solution.scanned_books.update(books_to_scan)
    
    def _recalculate_scanned_books(self, solution: Solution):
        """Recalculate all scanned books based on current library order"""
        solution.scanned_books = set()
        solution.scanned_books_per_library = {}
        
        curr_time = 0
        for lib_id in solution.signed_libraries:
            library = self.data.libs[lib_id]
            curr_time += library.signup_days
            
            if curr_time >= self.data.num_days:
                solution.signed_libraries = solution.signed_libraries[:solution.signed_libraries.index(lib_id)]
                break
                
            time_left = self.data.num_days - curr_time
            max_books = time_left * library.books_per_day
            
            available_books = [b.id for b in library.books if b.id not in solution.scanned_books]
            books_to_scan = sorted(available_books, key=lambda b: self.data.scores[b], reverse=True)[:max_books]
            
            if books_to_scan:
                solution.scanned_books_per_library[lib_id] = books_to_scan
                solution.scanned_books.update(books_to_scan) 