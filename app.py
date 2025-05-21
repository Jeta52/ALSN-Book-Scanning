import sys
import os
from models import Parser
from models.initial_solution import InitialSolution
from models.ALNS_solver import ALNSSolver
from validator.multiple_validator import validate_all_solutions

def run_instances(output_dir='output'):
    print(f"Output directory: {output_dir}")
    directory = os.listdir('input')
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for file in directory:
        if file.endswith('.txt'):
            print(f'\nProcessing instance: ./input/{file}')
            print('-' * 50)
            
            # Parse instance
            parser = Parser(f'./input/{file}')
            instance = parser.parse()
            
            # Create and run ALNS solver
            alns_solver = ALNSSolver(instance)
            solution = alns_solver.solve()  # Will run for 10 minutes
            
            # Save results
            score = solution.fitness_score
            results.append((file, score))
            
            # Export solution
            output_file = os.path.join(output_dir, file)
            solution.export(output_file)
            print(f"Solution exported to: {output_file}")
            print('-' * 50)

    print("\nValidating all solutions...")
    validate_all_solutions(input_dir='input', output_dir=output_dir)

    # Print summary of all instances
    print("\nSummary of all instances:")
    print("-" * 50)
    print(f"{'Instance':<30} {'Score':>15}")
    print("-" * 50)
    for file, score in results:
        print(f"{file:<30} {score:>15,}")
    print("-" * 50)

    # Write summary to a text file
    summary_file = os.path.join(output_dir, 'summary_results.txt')
    with open(summary_file, 'w') as f:
        f.write("Summary of all instances:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Instance':<30} {'Score':>15}\n")
        f.write("-" * 50 + "\n")
        for file, score in results:
            f.write(f"{file:<30} {score:>15,}\n")
        f.write("-" * 50 + "\n")
    print(f"\nSummary has been written to: {summary_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run ALNS solver for book scanning problem')
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Output directory for solutions')
    
    args = parser.parse_args()
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        subdir = sys.argv[1]
        run_instances(f"./output/{subdir}")
    else:
        print(f"Saving outputs to {args.output_dir}")
        run_instances(args.output_dir)


if __name__ == "__main__":
    main()