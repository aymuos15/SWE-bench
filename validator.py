
import pandas as pd
import textwrap
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

def load_dataset_file(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load a dataset from a file (JSON, JSONL, or Parquet).
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Tuple of (DataFrame, dataset_type)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data), 'json'
    elif file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data), 'jsonl'
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path), 'parquet'
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def load_original_swebench() -> pd.DataFrame:
    """Load the original SWE-bench dataset."""
    from datasets import load_dataset
    print("Loading original SWE-bench dataset from Hugging Face...")
    return pd.DataFrame(load_dataset('princeton-nlp/SWE-bench', split='test'))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Validate SWE-bench dataset')
parser.add_argument('--dataset', type=str, default=None,
                    help='Path to dataset file (JSON, JSONL, or Parquet). If not provided, uses the original SWE-bench dataset.')
args = parser.parse_args()

# Load the appropriate dataset
if args.dataset:
    print(f"Loading dataset from: {args.dataset}")
    df, dataset_type = load_dataset_file(args.dataset)
    print(f"Loaded {len(df)} examples from {dataset_type.upper()} file")
else:
    print("No dataset file provided. Using original SWE-bench dataset.")
    df = load_original_swebench()

# Get the first example for demonstration
if len(df) > 0:
    example = df.iloc[0]
else:
    print("Warning: The dataset is empty")
    example = None

def print_section(title, content, max_width=100):
    print(f"\n{'='*50}")
    print(f"{title.upper()}")
    print('-'*50)
    if isinstance(content, str):
        # Wrap long lines for better readability
        wrapped = textwrap.fill(content, width=max_width)
        print(wrapped)
    else:
        print(content)

# Display the example in a structured way
print_section("Repository & Version", f"{example['repo']} (Instance: {example['instance_id']})")
print_section("Base Commit", example['base_commit'])
print_section("Environment Setup Commit", example['environment_setup_commit'])

print_section("Problem Statement", example['problem_statement'])

if example['hints_text']:
    print_section("Hints", example['hints_text'])

print_section("Patch (Solution)", example['patch'])

if example['test_patch']:
    print_section("Test Modifications", example['test_patch'])

print_section("Tests to Fix (FAIL_TO_PASS)", example['FAIL_TO_PASS'])
print_section("Tests to Keep Passing (PASS_TO_PASS)", example['PASS_TO_PASS'])

def validate_dataset(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate the dataset for required fields and data integrity.
    
    Args:
        df: DataFrame containing the SWE-bench dataset
        
    Returns:
        Dict containing validation results and any errors found
    """
    results = {
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required fields
    required_fields = [
        'repo', 'instance_id', 'base_commit', 'problem_statement',
        'FAIL_TO_PASS', 'PASS_TO_PASS', 'environment_setup_commit'
    ]
    
    # Check for missing columns
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        results['status'] = 'FAIL'
        results['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check for empty values in required fields
    for field in required_fields:
        if field in df.columns and df[field].isnull().any():
            results['warnings'].append(f"Field '{field}' contains {df[field].isnull().sum()} null values")
    
    # Check for duplicate instance_ids
    duplicate_ids = df['instance_id'].duplicated()
    if duplicate_ids.any():
        results['status'] = 'FAIL'
        results['errors'].append(f"Found {duplicate_ids.sum()} duplicate instance_ids")
    
    # Gather basic statistics
    results['stats'] = {
        'total_examples': len(df),
        'unique_repositories': df['repo'].nunique(),
        'repositories_distribution': df['repo'].value_counts().to_dict()
    }
    
    return results

def print_example(example: Dict[str, Any]):
    """Print an example from the dataset."""
    if example is None:
        return
        
    print_section("Repository & Version", f"{example.get('repo', 'N/A')} (Instance: {example.get('instance_id', 'N/A')})")
    print_section("Base Commit", example.get('base_commit', 'N/A'))
    print_section("Environment Setup Commit", example.get('environment_setup_commit', 'N/A'))
    print_section("Problem Statement", example.get('problem_statement', 'N/A'))

    if example.get('hints_text'):
        print_section("Hints", example['hints_text'])

    if 'patch' in example:
        print_section("Patch (Solution)", example['patch'])

    if example.get('test_patch'):
        print_section("Test Modifications", example['test_patch'])

    if 'FAIL_TO_PASS' in example:
        print_section("Tests to Fix (FAIL_TO_PASS)", example['FAIL_TO_PASS'])
    
    if 'PASS_TO_PASS' in example:
        print_section("Tests to Keep Passing (PASS_TO_PASS)", example['PASS_TO_PASS'])

def main():
    # Print dataset info
    print(f"\n{'='*50}")
    print(f"DATASET VALIDATION")
    print(f"{'='*50}")
    print(f"Dataset source: {args.dataset or 'Original SWE-bench'}")
    print(f"Total examples: {len(df)}")
    
    # Show an example if available
    if example is not None:
        print("\nHere's an example from the dataset:")
        print_example(example)
    
    # Validate the dataset
    print("\n" + "="*50)
    print("VALIDATING DATASET")
    print("="*50)
    
    results = validate_dataset(df)
    
    # Print validation results
    print_section("Validation Results", results['status'])
    
    if results['errors']:
        print("\nErrors found:")
        for error in results['errors']:
            print(f"- {error}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"- {warning}")
    
    # Print dataset stats
    stats = results['stats']
    print_section("Dataset Statistics", 
                  f"Total examples: {stats['total_examples']}\n"
                  f"Number of unique repositories: {stats['unique_repositories']}")
    
    # Show repository distribution
    if stats['repositories_distribution']:
        print_section("Repository Distribution", 
                     "\n".join([f"- {k}: {v}" for k, v in stats['repositories_distribution'].items()]))
    
    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()

# example usage
# python validator.py --dataset data/extended_dataset/swebench_extended.parquet
# python validator.py #? For standard SWE-Bench