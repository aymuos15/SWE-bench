#!/usr/bin/env python3
"""
Creator script for analyzing valid tasks and extending the SWE-bench dataset.

This script:
1. Counts the number of valid tasks created for each repository
2. Creates a copy of the original SWE-bench dataset and appends valid tasks to it
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datasets import load_dataset

# Configuration
DATA_DIR = "data"
REPOS_DIR = os.path.join(DATA_DIR, "repos")
OUTPUT_DIR = os.path.join(DATA_DIR, "extended_dataset")


def load_swebench_dataset() -> pd.DataFrame:
    """Load the original SWE-bench dataset."""
    print("Loading original SWE-bench dataset...")
    dataset = load_dataset('princeton-nlp/SWE-bench', split='test')
    return pd.DataFrame(dataset)


def find_task_files() -> Dict[str, List[str]]:
    """
    Find all task JSON/JSONL files in the repository directories.
    
    Returns:
        Dict mapping repository names to lists of their task file paths
    """
    task_files = {}
    
    if not os.path.exists(REPOS_DIR):
        print(f"Warning: Repository directory not found: {REPOS_DIR}")
        return task_files
    
    for repo_dir in os.listdir(REPOS_DIR):
        repo_path = os.path.join(REPOS_DIR, repo_dir)
        tasks_dir = os.path.join(repo_path, "tasks")
        
        if not os.path.isdir(tasks_dir):
            continue
            
        repo_tasks = []
        for filename in os.listdir(tasks_dir):
            if filename.endswith(('.json', '.jsonl')):
                repo_tasks.append(os.path.join(tasks_dir, filename))
        
        if repo_tasks:
            task_files[repo_dir] = repo_tasks
    
    return task_files


def validate_task(task_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single task.
    
    Args:
        task_data: Dictionary containing task data
        
    Returns:
        Tuple of (is_valid, errors)
    """
    required_fields = [
        'repo', 
        'instance_id', 
        'base_commit', 
        'problem_statement',
    ]
    
    # These fields are required for SWE-bench compatibility
    # but might be missing in the generated tasks
    recommended_fields = [
        'FAIL_TO_PASS', 
        'PASS_TO_PASS', 
        'environment_setup_commit'
    ]
    
    errors = []
    warnings = []
    
    # Check for missing required fields
    for field in required_fields:
        if field not in task_data:
            errors.append(f"Missing required field: {field}")
        elif not task_data[field]:
            errors.append(f"Empty value for required field: {field}")
    
    # Check for recommended fields
    for field in recommended_fields:
        if field not in task_data or not task_data[field]:
            warnings.append(f"Missing recommended field: {field}")
    
    # Check for valid instance_id format (repo_name-PR#-instance#)
    if 'instance_id' in task_data and not isinstance(task_data['instance_id'], str):
        errors.append("instance_id must be a string")
    
    # If there are only warnings, still consider the task valid
    if not errors and warnings:
        print(f"  Warning: Task has missing recommended fields: {', '.join(warnings)}")
        return True, []
    
    return len(errors) == 0, errors


def process_tasks() -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Process all task files and validate them.
    
    Returns:
        Tuple of (stats, valid_tasks) where:
            - stats: Dictionary with counts of total and valid tasks per repo
            - valid_tasks: List of all valid task dictionaries
    """
    task_files = find_task_files()
    stats = {}
    all_valid_tasks = []
    
    print(f"Found task files for {len(task_files)} repositories")
    
    for repo, files in task_files.items():
        print(f"\nProcessing repository: {repo} ({len(files)} files)")
        
        valid_count = 0
        total_tasks = 0
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    if file_path.endswith('.jsonl'):
                        # Process JSONL file (one JSON object per line)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                                
                            total_tasks += 1
                            try:
                                task_data = json.loads(line)
                                is_valid, errors = validate_task(task_data)
                                
                                if is_valid:
                                    valid_count += 1
                                    all_valid_tasks.append(task_data)
                                else:
                                    print(f"  Invalid task in {os.path.basename(file_path)}: {', '.join(errors)}")
                                    
                            except json.JSONDecodeError as e:
                                print(f"  Error parsing JSON in {os.path.basename(file_path)}: {str(e)}")
                    else:
                        # Process regular JSON file
                        task_data = json.load(f)
                        total_tasks += 1
                        
                        is_valid, errors = validate_task(task_data)
                        
                        if is_valid:
                            valid_count += 1
                            all_valid_tasks.append(task_data)
                        else:
                            print(f"  Invalid task in {os.path.basename(file_path)}: {', '.join(errors)}")
                    
            except IOError as e:
                print(f"  Error reading {file_path}: {str(e)}")
        
        # If we processed JSONL files, use the actual task count
        # Otherwise, use the number of files as the task count
        total = total_tasks if total_tasks > 0 else len(files)
        
        stats[repo] = {
            'total_tasks': total,
            'valid_tasks': valid_count
        }
        print(f"  Valid tasks: {valid_count}/{total}")
    
    return stats, all_valid_tasks


def extend_swebench_dataset(original_df: pd.DataFrame, new_tasks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extend the original SWE-bench dataset with new valid tasks.
    
    Args:
        original_df: DataFrame of the original SWE-bench dataset
        new_tasks: List of new valid task dictionaries to add
        
    Returns:
        Extended DataFrame with original and new tasks
    """
    if not new_tasks:
        print("No valid tasks to add to the dataset")
        return original_df
    
    # Create a DataFrame from new tasks
    new_df = pd.DataFrame(new_tasks)
    
    # Ensure all required columns exist in the new DataFrame
    for col in original_df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    
    # Reorder columns to match the original DataFrame
    new_df = new_df[original_df.columns]
    
    # Concatenate original and new DataFrames
    extended_df = pd.concat([original_df, new_df], ignore_index=True)
    
    # Check for duplicate instance_ids
    duplicate_ids = extended_df['instance_id'].duplicated()
    if duplicate_ids.any():
        print(f"Warning: Found {duplicate_ids.sum()} duplicate instance_ids in the extended dataset")
    
    return extended_df


def save_extended_dataset(df: pd.DataFrame, output_dir: str):
    """Save the extended dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "swebench_extended.parquet")
    
    # Save as parquet for efficiency
    df.to_parquet(output_path, index=False)
    
    # Also save as JSON for compatibility
    json_path = os.path.join(output_dir, "swebench_extended.json")
    df.to_json(json_path, orient='records', indent=2)
    
    print(f"\nExtended dataset saved to:"
          f"\n- {output_path}"
          f"\n- {json_path}")


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process tasks and get statistics
    stats, valid_tasks = process_tasks()
    
    # Print summary
    print("\n" + "="*50)
    print("TASK PROCESSING SUMMARY")
    print("="*50)
    
    total_tasks = sum(repo_stats['total_tasks'] for repo_stats in stats.values())
    total_valid = sum(repo_stats['valid_tasks'] for repo_stats in stats.values())
    
    print(f"\nTotal repositories: {len(stats)}")
    print(f"Total tasks processed: {total_tasks}")
    
    if total_tasks > 0:
        print(f"Total valid tasks: {total_valid} ({total_valid/total_tasks*100:.1f}%)")
        
        print("\nRepository details:")
        for repo, repo_stats in stats.items():
            if repo_stats['total_tasks'] > 0:  # Avoid division by zero
                valid_pct = (repo_stats['valid_tasks'] / repo_stats['total_tasks']) * 100
                print(f"- {repo}: {repo_stats['valid_tasks']}/{repo_stats['total_tasks']} valid ({valid_pct:.1f}%)")
    else:
        print("No tasks found in any repository.")
    
    # Only proceed if we have valid tasks
    if not valid_tasks:
        print("\nNo valid tasks found. Exiting.")
        return
    
    # Load original dataset
    original_df = load_swebench_dataset()
    
    # Extend the dataset with new valid tasks
    extended_df = extend_swebench_dataset(original_df, valid_tasks)
    
    # Save the extended dataset
    save_extended_dataset(extended_df, OUTPUT_DIR)
    
    print("\nDataset extension complete!")


if __name__ == "__main__":
    main()
