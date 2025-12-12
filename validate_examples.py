"""
Code example validation script
This script validates that all code examples execute without errors
"""
import os
import subprocess
import sys
from pathlib import Path

def validate_python_examples():
    """Validate that Python examples execute without errors"""
    examples_dir = Path("examples")
    results = []
    
    for root, dirs, files in os.walk(examples_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Validating: {file_path}")
                
                try:
                    # Run the Python file with a timeout
                    result = subprocess.run(
                        [sys.executable, file_path], 
                        timeout=10,  # 10 second timeout
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        status = "PASS"
                    else:
                        status = "FAIL"
                        print(f"  Error: {result.stderr}")
                    
                    results.append((file_path, status))
                    print(f"  Status: {status}")
                    
                except subprocess.TimeoutExpired:
                    results.append((file_path, "TIMEOUT"))
                    print(f"  Status: TIMEOUT")
                except Exception as e:
                    results.append((file_path, f"ERROR: {str(e)}"))
                    print(f"  Status: ERROR: {str(e)}")
    
    return results

def validate_notebooks():
    """Validate Jupyter notebooks (basic check)"""
    notebooks_dir = Path("notebooks")
    results = []
    
    for root, dirs, files in os.walk(notebooks_dir):
        for file in files:
            if file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                print(f"Validating notebook: {file_path}")
                
                try:
                    # Just check if the notebook file is valid JSON
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook = json.load(f)
                    
                    if 'cells' in notebook and isinstance(notebook['cells'], list):
                        status = "PASS (Valid JSON structure)"
                    else:
                        status = "FAIL (Invalid notebook structure)"
                    
                    results.append((file_path, status))
                    print(f"  Status: {status}")
                    
                except Exception as e:
                    results.append((file_path, f"ERROR: {str(e)}"))
                    print(f"  Status: ERROR: {str(e)}")
    
    return results

def main():
    print("Validating code examples...")
    
    # Validate Python examples
    python_results = validate_python_examples()
    
    # Validate notebooks
    notebook_results = validate_notebooks()
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    print("\nPython Examples:")
    for file_path, status in python_results:
        print(f"  {status}: {file_path}")
    
    print("\nNotebooks:")
    for file_path, status in notebook_results:
        print(f"  {status}: {file_path}")
    
    # Count passes and failures
    python_passes = sum(1 for _, status in python_results if status == "PASS")
    python_total = len(python_results)
    
    notebook_passes = sum(1 for _, status in notebook_results if status.startswith("PASS"))
    notebook_total = len(notebook_results)
    
    print(f"\nPython Examples: {python_passes}/{python_total} passed")
    print(f"Notebooks: {notebook_passes}/{notebook_total} passed")
    
    total_passes = python_passes + notebook_passes
    total_items = python_total + notebook_total
    
    print(f"Overall: {total_passes}/{total_items} passed")
    
    # Return appropriate exit code
    if total_passes == total_items:
        print("\nAll validations passed!")
        return 0
    else:
        print(f"\n{total_items - total_passes} items failed validation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())