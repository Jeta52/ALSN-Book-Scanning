import os
from validator.validator import validate_solution

def validate_all_solutions(input_dir='input', output_dir='output'):
    print("\n=== Validating All Solutions ===")
    
    # Get all input files
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not input_files:
        print(f"No input files found in {input_dir}")
        return
    
    # Track validation results
    valid_count = 0
    invalid_count = 0
    
    # Validate each input/output pair
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file)
        
        print(f"\nValidating {input_file}...")
        
        # Check if output file exists
        if not os.path.exists(output_path):
            print(f"  ✗ No output file found")
            invalid_count += 1
            continue
        
        # Validate the solution
        result = validate_solution(input_path, output_path, isConsoleApplication=True)
        if result == "Valid":
            print(f"  ✓ Valid")
            valid_count += 1
        else:
            print(f"  ✗ Invalid")
            invalid_count += 1
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Total files checked: {len(input_files)}")
    print(f"Valid solutions: {valid_count}")
    print(f"Invalid solutions: {invalid_count}")
    
    return valid_count == len(input_files)

if __name__ == "__main__":
    validate_all_solutions() 