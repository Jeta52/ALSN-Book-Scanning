import random
import copy
from models.solution import Solution

class Tweaks:
    # Define weights for each tweak method
    WEIGHTS = {
        'swap_signed': 1.0,
        'swap_signed_with_unsigned': 3.0,
        'swap_same_books': 1.0,
        'swap_last_book': 1.0,
        'swap_neighbor_libraries': 1.0,
        'insert_library': 2.0,
        'crossover': 1.0
    }

    @staticmethod
    def get_tweak_methods():
        """Return list of tweak methods with their weights"""
        return [
            (Tweaks.tweak_solution_swap_signed, Tweaks.WEIGHTS['swap_signed']),
            (Tweaks.tweak_solution_swap_signed_with_unsigned, Tweaks.WEIGHTS['swap_signed_with_unsigned']),
            (Tweaks.tweak_solution_swap_same_books, Tweaks.WEIGHTS['swap_same_books']),
            (Tweaks.tweak_solution_swap_neighbor_libraries, Tweaks.WEIGHTS['swap_neighbor_libraries']),
            (Tweaks.tweak_solution_insert_library, Tweaks.WEIGHTS['insert_library']),
            (Tweaks.tweak_solution_crossover, Tweaks.WEIGHTS['crossover']),
            (Tweaks.tweak_solution_swap_last_book, Tweaks.WEIGHTS['swap_last_book'])
        ]

    @staticmethod
    def choose_tweak_method():
        """Randomly choose a tweak method based on weights"""
        methods, weights = zip(*Tweaks.get_tweak_methods())
        return random.choices(methods, weights=weights, k=1)[0]

    @staticmethod
    def tweak_solution_swap_signed(solution, data):
        """
        Randomly swaps two libraries within the signed libraries list.
        This creates a new solution by exchanging the positions of two libraries
        while maintaining the feasibility of the solution.

        Args:
            solution: The current solution to tweak
            data: The problem data

        Returns:
            A new solution with two libraries swapped
        """
        if len(solution.signed_libraries) < 2:
            return solution

        # Create a copy of the solution
        new_solution = Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )

        # Select two random libraries to swap
        idx1, idx2 = random.sample(range(len(new_solution.signed_libraries)), 2)
        new_solution.signed_libraries[idx1], new_solution.signed_libraries[idx2] = \
            new_solution.signed_libraries[idx2], new_solution.signed_libraries[idx1]

        # Rebuild the solution
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []

        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    @staticmethod
    def tweak_solution_swap_signed_with_unsigned(solution, data, bias_type=None, bias_ratio=2/3):
        if not solution.signed_libraries or not solution.unsigned_libraries:
            return solution

        # Create a copy of the solution
        new_solution = Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )

        total_signed = len(new_solution.signed_libraries)

        # Select signed library based on bias
        if bias_type == "favor_first_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(0, total_signed // 2 - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        elif bias_type == "favor_second_half":
            if random.random() < bias_ratio:
                signed_idx = random.randint(total_signed // 2, total_signed - 1)
            else:
                signed_idx = random.randint(0, total_signed - 1)
        else:
            signed_idx = random.randint(0, total_signed - 1)

        # Select unsigned library
        unsigned_idx = random.randint(0, len(new_solution.unsigned_libraries) - 1)

        # Swap libraries
        signed_lib_id = new_solution.signed_libraries[signed_idx]
        unsigned_lib_id = new_solution.unsigned_libraries[unsigned_idx]
        new_solution.signed_libraries[signed_idx] = unsigned_lib_id
        new_solution.unsigned_libraries[unsigned_idx] = signed_lib_id

        # Rebuild the solution
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []

        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    @staticmethod
    def tweak_solution_swap_same_books(solution, data):
        if len(solution.signed_libraries) < 2:
            return solution

        # Create a copy of the solution
        new_solution = Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )

        # Select two random libraries to swap
        idx1 = random.randint(0, len(new_solution.signed_libraries) - 1)
        idx2 = random.randint(0, len(new_solution.signed_libraries) - 1)
        while idx1 == idx2:
            idx2 = random.randint(0, len(new_solution.signed_libraries) - 1)

        # Swap the libraries
        new_solution.signed_libraries[idx1], new_solution.signed_libraries[idx2] = \
            new_solution.signed_libraries[idx2], new_solution.signed_libraries[idx1]

        # Rebuild the solution
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []

        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    @staticmethod
    def tweak_solution_swap_last_book(solution, data):
        if not solution.signed_libraries:
            return solution

        new_solution = Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )

        lib_id = random.choice(new_solution.signed_libraries)
        library = data.libs[lib_id]
        scanned_books = new_solution.scanned_books_per_library.get(lib_id, [])

        if not scanned_books:
            return new_solution

        last_book = scanned_books[-1]
        available_books = sorted(
            {book.id for book in library.books} - new_solution.scanned_books,
            key=lambda b: -data.scores[b]
        )

        if not available_books:
            return new_solution

        new_book = available_books[0]
        new_scanned_books = scanned_books[:-1] + [new_book]
        new_solution.scanned_books_per_library[lib_id] = new_scanned_books
        new_solution.scanned_books.remove(last_book)
        new_solution.scanned_books.add(new_book)

        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []

        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            if curr_time + library.signup_days >= data.num_days - 1:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    @staticmethod
    def tweak_solution_crossover(solution, data):
        """
        Performs crossover by:
        1. Randomly selecting a crossover point
        2. Creating two new solutions by combining parts of the original solution
        3. Selecting the better of the two solutions
        """
        if len(solution.signed_libraries) < 2:
            return solution

        # Select a random crossover point
        crossover_point = random.randint(1, len(solution.signed_libraries) - 1)
        
        # Create two new solutions
        solution1 = Solution(
            solution.signed_libraries[:crossover_point],
            solution.unsigned_libraries + solution.signed_libraries[crossover_point:],
            {},
            set()
        )
        
        solution2 = Solution(
            solution.signed_libraries[crossover_point:],
            solution.unsigned_libraries + solution.signed_libraries[:crossover_point],
            {},
            set()
        )
        
        # Rebuild both solutions
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []
        
        # Process solution1
        for lib_id in solution1.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                solution1.unsigned_libraries.append(lib_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                solution1.unsigned_libraries.append(lib_id)
        
        solution1.signed_libraries = new_signed_libraries
        solution1.scanned_books_per_library = new_scanned_books_per_library
        solution1.scanned_books = new_scanned_books
        solution1.calculate_fitness_score(data.scores)
        
        # Process solution2
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []
        
        for lib_id in solution2.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                solution2.unsigned_libraries.append(lib_id)
                continue
                
            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day
            
            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]
            
            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                solution2.unsigned_libraries.append(lib_id)
        
        solution2.signed_libraries = new_signed_libraries
        solution2.scanned_books_per_library = new_scanned_books_per_library
        solution2.scanned_books = new_scanned_books
        solution2.calculate_fitness_score(data.scores)
        
        # Return the better solution
        return solution1 if solution1.fitness_score > solution2.fitness_score else solution2

    @staticmethod
    def tweak_solution_swap_neighbor_libraries(solution, data):
        if len(solution.signed_libraries) < 2:
            return solution

        # Create a copy of the solution
        new_solution = Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )

        # Select a random position and its neighbor
        pos = random.randint(0, len(new_solution.signed_libraries) - 2)
        new_solution.signed_libraries[pos], new_solution.signed_libraries[pos + 1] = \
            new_solution.signed_libraries[pos + 1], new_solution.signed_libraries[pos]

        # Rebuild the solution
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []

        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution

    @staticmethod
    def tweak_solution_insert_library(solution, data):
        if not solution.unsigned_libraries:
            return solution

        # Create a copy of the solution
        new_solution = Solution(
            solution.signed_libraries.copy(),
            solution.unsigned_libraries.copy(),
            solution.scanned_books_per_library.copy(),
            solution.scanned_books.copy()
        )

        # Select a random unsigned library
        unsigned_idx = random.randint(0, len(new_solution.unsigned_libraries) - 1)
        new_lib_id = new_solution.unsigned_libraries.pop(unsigned_idx)

        # Select a random position to insert
        insert_pos = random.randint(0, len(new_solution.signed_libraries))
        new_solution.signed_libraries.insert(insert_pos, new_lib_id)

        # Rebuild the solution
        curr_time = 0
        new_scanned_books = set()
        new_scanned_books_per_library = {}
        new_signed_libraries = []

        for lib_id in new_solution.signed_libraries:
            library = data.libs[lib_id]
            
            # Check if there's enough time for signup and at least one day of scanning
            if curr_time + library.signup_days >= data.num_days - 1:
                new_solution.unsigned_libraries.append(lib_id)
                continue

            time_left = data.num_days - (curr_time + library.signup_days)
            max_books_scanned = time_left * library.books_per_day

            available_books = sorted(
                {book.id for book in library.books} - new_scanned_books,
                key=lambda b: -data.scores[b]
            )[:max_books_scanned]

            if available_books:
                new_signed_libraries.append(lib_id)
                new_scanned_books_per_library[lib_id] = available_books
                new_scanned_books.update(available_books)
                curr_time += library.signup_days
            else:
                new_solution.unsigned_libraries.append(lib_id)

        new_solution.signed_libraries = new_signed_libraries
        new_solution.scanned_books_per_library = new_scanned_books_per_library
        new_solution.scanned_books = new_scanned_books
        new_solution.calculate_fitness_score(data.scores)

        return new_solution 