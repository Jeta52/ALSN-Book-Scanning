import random
import time
import heapq
from models.solution import Solution
from models.library import Library
from models.local_search import LocalSearch


class InitialSolution:

    @staticmethod
    def generate_initial_solution_grasp(data, p=0.05, max_time=60):
        start_time = time.time()
        best_solution = None
        Library._id_counter = 0

        while time.time() - start_time < max_time:
            candidate_solution = InitialSolution.build_grasp_solution(data, p)

            improved_solution = LocalSearch.local_search(
                candidate_solution, data, time_limit=5, max_iterations=100
            )

            if (best_solution is None) or (
                improved_solution.fitness_score > best_solution.fitness_score
            ):
                best_solution = improved_solution

        return best_solution

    @staticmethod
    def build_grasp_solution(data, p=0.05):
        libs_sorted = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books)),
        )

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        candidate_libs = libs_sorted[:]

        while candidate_libs:
            rcl_size = max(1, int(len(candidate_libs) * p))
            rcl = candidate_libs[:rcl_size]

            chosen_lib = random.choice(rcl)
            candidate_libs.remove(chosen_lib)

            if curr_time + chosen_lib.signup_days >= data.num_days:
                unsigned_libraries.append(chosen_lib.id)
            else:
                time_left = data.num_days - (curr_time + chosen_lib.signup_days)
                max_books_scanned = time_left * chosen_lib.books_per_day

                available_books = sorted(
                    {book.id for book in chosen_lib.books} - scanned_books,
                    key=lambda b: -data.scores[b],
                )[:max_books_scanned]

                if available_books:
                    signed_libraries.append(chosen_lib.id)
                    scanned_books_per_library[chosen_lib.id] = available_books
                    scanned_books.update(available_books)
                    curr_time += chosen_lib.signup_days
                else:
                    unsigned_libraries.append(chosen_lib.id)

        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        solution.calculate_fitness_score(data.scores)
        return solution

    @staticmethod
    def generate_initial_solution_sorted(data):
        Library._id_counter = 0
        sorted_libraries = sorted(
            data.libs,
            key=lambda l: (l.signup_days, -sum(data.scores[b.id] for b in l.books)),
        )

        signed_libraries = []
        unsigned_libraries = []
        scanned_books_per_library = {}
        scanned_books = set()
        curr_time = 0

        for library in sorted_libraries:
            if curr_time + library.signup_days >= data.num_days:
                unsigned_libraries.append(library.id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - scanned_books,
                key=lambda b: -data.scores[b],
            )[:max_books_scanned]

            if available_books:
                signed_libraries.append(library.id)
                scanned_books_per_library[library.id] = available_books
                scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                unsigned_libraries.append(library.id)

        solution = Solution(
            signed_libraries,
            unsigned_libraries,
            scanned_books_per_library,
            scanned_books,
        )
        solution.calculate_fitness_score(data.scores)

        return solution

    @staticmethod
    def generate_initial_solution_greedy(data):
        Library._id_counter = 0
        lib_info = []
        for lib in data.libs:
            sorted_books = sorted(lib.books, key=lambda b: -data.scores[b.id])
            total_score = sum(data.scores[b.id] for b in sorted_books)
            lib_info.append(
                {"lib": lib, "sorted_books": sorted_books, "total_score": total_score}
            )

        heap = []
        for idx, info in enumerate(lib_info):
            lib = info["lib"]
            if lib.signup_days < data.num_days:
                max_books = (data.num_days - lib.signup_days) * lib.books_per_day
                score = sum(data.scores[b.id] for b in info["sorted_books"][:max_books])
                efficiency = (
                    score / lib.signup_days if lib.signup_days > 0 else float("inf")
                )
                heapq.heappush(heap, (-efficiency, idx))

        signed = []
        scanned_books = set()
        scanned_per_lib = {}
        curr_time = 0
        used_libs = set()

        while heap and curr_time < data.num_days:
            _, idx = heapq.heappop(heap)
            if idx in used_libs:
                continue
            info = lib_info[idx]
            lib = info["lib"]
            if curr_time + lib.signup_days >= data.num_days:
                continue
            time_left = data.num_days - (curr_time + lib.signup_days)
            max_books = time_left * lib.books_per_day
            available_books = [
                b.id for b in info["sorted_books"] if b.id not in scanned_books
            ][:max_books]
            if not available_books:
                continue
            signed.append(lib.id)
            scanned_per_lib[lib.id] = available_books
            scanned_books.update(available_books)
            curr_time += lib.signup_days
            used_libs.add(idx)

        sol = Solution(signed, [], scanned_per_lib, scanned_books)
        sol.calculate_fitness_score(data.scores)
        return sol

    @staticmethod
    def generate_initial_solution_weighted_efficiency(data, alpha=1, beta=0.1):
        Library._id_counter = 0
        libs = data.libs[:]
        curr_time = 0
        scanned_books = set()
        scanned_per_lib = {}
        signed_libs = []
        unsigned_libs = []

        used = 0
        while libs and curr_time < data.num_days:
            lib_scores = []
            for lib in libs:
                if curr_time + lib.signup_days >= data.num_days:
                    continue
                time_left = data.num_days - (curr_time + lib.signup_days)
                max_books = time_left * lib.books_per_day
                books = sorted(
                    [b.id for b in lib.books if b.id not in scanned_books],
                    key=lambda x: -data.scores[x],
                )[:max_books]
                score = sum(data.scores[b] for b in books)
                if score:
                    penalty = (lib.signup_days**alpha) * (1 + beta * used)
                    lib_scores.append((score / penalty, lib, books))

            if not lib_scores:
                break

            _, best_lib, best_books = max(lib_scores, key=lambda x: x[0])
            libs.remove(best_lib)
            signed_libs.append(best_lib.id)
            scanned_per_lib[best_lib.id] = best_books
            scanned_books.update(best_books)
            curr_time += best_lib.signup_days
            used += 1

        sol = Solution(signed_libs, unsigned_libs, scanned_per_lib, scanned_books)
        sol.calculate_fitness_score(data.scores)
        return sol

    @staticmethod
    def tune_weighted_efficiency_parameters(data, time_limit=60):
        start_time = time.time()
        best_score = 0
        best_alpha = 1.0
        best_beta = 0.1
        best_solution = None

        alpha_values = [1.0, 0.5,  1.5, 2.0]
        beta_values = [0.0, 0.05, 0.1, 0.2]

        for alpha in alpha_values:
            for beta in beta_values:
                if time.time() - start_time >= time_limit:
                    return best_alpha, best_beta, best_score, best_solution

                solution = (
                    InitialSolution.generate_initial_solution_weighted_efficiency(
                        data, alpha=alpha, beta=beta
                    )
                )
                score = solution.fitness_score

                if score > best_score:
                    best_score = score
                    best_alpha = alpha
                    best_beta = beta
                    best_solution = solution

        return best_alpha, best_beta, best_score, best_solution

    @staticmethod
    def generate_initial_greedy_heap(data):
        book_scores = data.scores
        
        lib_info = []
        for lib_id, lib in enumerate(data.libs):
            sorted_books = sorted(lib.books, key=lambda b: -book_scores[b.id])
            total_score = sum(book_scores[b.id] for b in sorted_books)
            
            if lib.signup_days < data.num_days:
                max_scannable = (data.num_days - lib.signup_days) * lib.books_per_day
                potential_score = sum(book_scores[b.id] for b in sorted_books[:min(max_scannable, len(sorted_books))])
                efficiency = potential_score / lib.signup_days if lib.signup_days > 0 else float('inf')
            else:
                efficiency = 0
                
            lib_info.append({
                'id': lib_id,
                'lib': lib,
                'sorted_books': sorted_books,
                'total_score': total_score,
                'efficiency': efficiency
            })
        
        heap = []
        for info in lib_info:
            if info['efficiency'] > 0:
                heapq.heappush(heap, (-info['efficiency'], info['id']))
        
        scanned_books = set()
        current_day = 0
        used_libs = set()
        
        signed_libs = []
        unsigned_libs = []
        scanned_books_per_library = {}
        
        while heap and current_day < data.num_days:
            _, lib_id = heapq.heappop(heap)
            
            if lib_id in used_libs:
                continue
                
            lib = data.libs[lib_id]
            info = lib_info[lib_id]
            
            if current_day + lib.signup_days >= data.num_days:
                unsigned_libs.append(lib_id)
                continue
                
            days_left = data.num_days - (current_day + lib.signup_days)
            max_scannable = min(days_left * lib.books_per_day, len(lib.books))
            
            available_books = [b.id for b in info['sorted_books'] if b.id not in scanned_books]
            
            if not available_books:
                unsigned_libs.append(lib_id)
                continue
                
            books_to_scan = available_books[:max_scannable]
            
            if not books_to_scan:
                unsigned_libs.append(lib_id)
                continue
                
            signed_libs.append(lib_id)
            scanned_books_per_library[lib_id] = books_to_scan
            
            scanned_books.update(books_to_scan)
            
            current_day += lib.signup_days
            
            used_libs.add(lib_id)
            
            if len(heap) > 0 and len(heap) % 1000 == 0:
                new_heap = []
                for neg_eff, lid in heap:
                    if lid in used_libs:
                        continue
                        
                    l = data.libs[lid]
                    info = lib_info[lid]
                    
                    if current_day + l.signup_days >= data.num_days:
                        continue
                        
                    days_left = data.num_days - (current_day + l.signup_days)
                    max_scannable = min(days_left * l.books_per_day, len(l.books))
                    
                    unscanned = [b.id for b in info['sorted_books'] if b.id not in scanned_books]
                    potential = sum(book_scores[b] for b in unscanned[:max_scannable])
                    
                    efficiency = potential / l.signup_days if l.signup_days > 0 else float('inf')
                    if efficiency > 0:
                        heapq.heappush(new_heap, (-efficiency, lid))
                        
                heap = new_heap
        
        for lib_id in range(len(data.libs)):
            if lib_id not in used_libs:
                unsigned_libs.append(lib_id)
        
        solution = Solution(
            signed_libs,
            unsigned_libs,
            scanned_books_per_library,
            scanned_books
        )
        
        solution.calculate_fitness_score(data.scores)
        
        return solution

    @staticmethod
    def generate_initial_solution(data):
        best_solution = None
        print("\nGenerating solutions using different methods:")
        print("-" * 50)

        best_alpha, best_beta, best_score, weighted_solution = (
            InitialSolution.tune_weighted_efficiency_parameters(data, time_limit=60)
        )
        print(f"\nWeighted Efficiency Solution:")
        print(f"Score: {weighted_solution.fitness_score}")
        print(f"Best Alpha: {best_alpha}, Best Beta: {best_beta}")
        best_solution = weighted_solution

        generation_methods = [
           
            (
                InitialSolution.generate_initial_greedy_heap,
                {},
                "Greedy"
            ),
           
            (
                InitialSolution.generate_initial_solution_grasp,
                {"p": 0.03, "max_time": 15},
                "GRASP"
            ),
            (
                InitialSolution.generate_initial_solution_sorted,
                {},
                "Sorted"
            ),
        ]

        for method, kwargs, method_name in generation_methods:
            try:
                initial_solution = method(data, **kwargs)
                print(f"\n{method_name} Solution:")
                print(f"Score: {initial_solution.fitness_score}")
                if initial_solution.fitness_score > 0:
                    if (
                        best_solution is None
                        or initial_solution.fitness_score > best_solution.fitness_score
                    ):
                        best_solution = initial_solution
            except Exception as e:
                print(f"Error generating solution with {method_name}: {e}")
                continue

        print("\nBest Solution Found:")
        print(f"Score: {best_solution.fitness_score}")
        print("-" * 50)

        if best_solution is None:
            raise Exception("No valid initial solution could be generated")

        return best_solution
