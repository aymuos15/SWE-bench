#!/usr/bin/env python3
"""
Docker Setup Verification Script for SWE-bench

This script:
1. Lists all unique repositories in a dataset
2. Selects 3 repositories (configurable)
3. Verifies that Docker images can be built and containers work properly

Usage:
    python verify_docker_setups.py --dataset data/extended_dataset/swebench_extended.parquet
    python verify_docker_setups.py --dataset data/extended_dataset/swebench_extended.json --num-repos 5
    python verify_docker_setups.py  # Uses original SWE-bench dataset
"""

import argparse
import docker
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

# Import SWE-bench functionality
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.constants import KEY_INSTANCE_ID
from validator import load_dataset_file


class DockerVerificationError(Exception):
    """Custom exception for Docker verification failures"""
    pass


def print_section(title: str, content: str = "", separator: str = "="):
    """Print a formatted section header"""
    print(f"\n{separator * 60}")
    print(f"{title.upper()}")
    print(f"{separator * 60}")
    if content:
        print(content)


def get_unique_repos(dataset: List[Dict]) -> Dict[str, int]:
    """
    Extract unique repositories from the dataset and count instances per repo.
    
    Args:
        dataset: List of dataset instances
        
    Returns:
        Dictionary mapping repo names to instance counts
    """
    repo_counts = Counter()
    for instance in dataset:
        repo = instance.get('repo', 'unknown')
        repo_counts[repo] += 1
    
    return dict(repo_counts)


def select_repos_for_verification(repo_counts: Dict[str, int], num_repos: int = None) -> List[str]:
    """
    Select repositories for verification, optionally limiting to a specific number.
    
    Args:
        repo_counts: Dictionary mapping repo names to instance counts
        num_repos: Maximum number of repositories to select (None for all)
        
    Returns:
        List of selected repository names
    """
    # Sort repositories by instance count (descending)
    sorted_repos = sorted(repo_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Apply num_repos limit if specified
    if num_repos is not None and num_repos > 0:
        sorted_repos = sorted_repos[:num_repos]
    
    # Extract just the repository names
    selected_repos = [repo for repo, _ in sorted_repos]
    
    return selected_repos


def filter_dataset_by_repos(dataset: List[Dict], selected_repos: List[str]) -> List[Dict]:
    """
    Filter dataset to only include instances from selected repositories.
    
    Args:
        dataset: Full dataset
        selected_repos: List of repository names to include
        
    Returns:
        Filtered dataset containing only instances from selected repos
    """
    filtered = []
    for instance in dataset:
        if instance.get('repo') in selected_repos:
            filtered.append(instance)
    
    print(f"Filtered dataset: {len(filtered)} instances from {len(selected_repos)} repositories")
    return filtered

def print_verification_summary(all_results: List[Dict[str, any]]):
    """Print a summary of all verification results"""
    print_section("Docker Verification Summary")
    
    total_repos = len(all_results)
    total_successful_builds = sum(r['successful_builds'] for r in all_results)
    total_failed_builds = sum(r['failed_builds'] for r in all_results)
    total_successful_containers = sum(r['successful_containers'] for r in all_results)
    total_failed_containers = sum(r['failed_containers'] for r in all_results)
    total_errors = sum(len(r['errors']) for r in all_results)
    
    print(f"Repositories tested: {total_repos}")
    print(f"Image builds - Success: {total_successful_builds}, Failed: {total_failed_builds}")
    print(f"Container tests - Success: {total_successful_containers}, Failed: {total_failed_containers}")
    print(f"Total errors encountered: {total_errors}")
    
    # Print per-repo summary
    print(f"\nPer-repository results:")
    for result in all_results:
        status = "✓ PASS" if result['successful_containers'] > 0 and len(result['errors']) == 0 else "✗ FAIL"
        print(f"  {result['repo']}: {status}")
        print(f"    - Builds: {result['successful_builds']}/{result['successful_builds'] + result['failed_builds']}")
        print(f"    - Containers: {result['successful_containers']}/{result['successful_containers'] + result['failed_containers']}")
        if result['errors']:
            print(f"    - Errors: {len(result['errors'])}")
    
    # Overall status
    overall_success = total_successful_containers > 0 and total_errors == 0
    print(f"\n{'='*60}")
    print(f"OVERALL STATUS: {'✓ VERIFICATION PASSED' if overall_success else '✗ VERIFICATION FAILED'}")
    print(f"{'='*60}")


def list_all_repositories(repo_counts):
    """List all repositories with their instance counts."""
    print("\n" + "="*80)
    print("AVAILABLE REPOSITORIES")
    print("="*80)
    
    # Sort repositories by name for consistent ordering
    sorted_repos = sorted(repo_counts.items(), key=lambda x: x[0].lower())
    
    # Print repositories in columns for better readability
    max_len = max(len(repo) for repo, _ in sorted_repos) + 2
    for i, (repo, count) in enumerate(sorted_repos, 1):
        print(f"{i:3d}. {repo:{max_len}} ({count} instances)")
    print()

def interactive_select_repos(repo_counts):
    """Interactively select repositories to verify."""
    # Convert to list of (repo, count) tuples sorted by name for consistent ordering
    # This should match the order shown in list_all_repositories
    repos = sorted(repo_counts.items(), key=lambda x: x[0].lower())
    
    while True:
        print("\nEnter repository numbers or names (comma/space separated):")
        print("Examples: '1,3,5' or 'django/requests scipy/scipy'")
        print("Enter 'all' to select all repositories")
        print("Enter 'q' to finish selection")
        
        user_input = input("> ").strip()
        
        if user_input.lower() == 'q':
            return []
            
        if user_input.lower() == 'all':
            return [repo for repo, _ in repos]
        
        # Split input by commas or spaces
        selections = [s.strip() for s in user_input.replace(',', ' ').split()]
        
        if not selections:
            print("Please enter at least one repository")
            continue
            
        selected_repos = []
        invalid_selections = []
        
        for sel in selections:
            # Check if input is a number
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(repos):
                    selected_repos.append(repos[idx][0])
                else:
                    invalid_selections.append(sel)
            else:
                # Treat as repository name
                if sel in repo_counts:
                    selected_repos.append(sel)
                else:
                    invalid_selections.append(sel)
        
        # Remove duplicates while preserving order
        selected_repos = list(dict.fromkeys(selected_repos))
        
        if invalid_selections:
            print(f"\nWarning: Invalid selections - {' '.join(invalid_selections)}")
            print("Please try again or enter 'q' to cancel")
            continue
            
        if not selected_repos:
            print("No valid repositories selected. Please try again.")
            continue
            
        # Show selection and confirm
        print("\nSelected repositories:")
        for repo in selected_repos:
            print(f"  - {repo} ({repo_counts[repo]} instances)")
            
        confirm = input("\nIs this correct? [Y/n] ").strip().lower()
        if not confirm or confirm == 'y':
            return selected_repos

def main():
    """Main function to orchestrate the Docker verification process"""
    parser = argparse.ArgumentParser(
        description='Verify Docker setups for SWE-bench repositories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Verify using original SWE-bench dataset
    python swebench_verify_docker.py
    
    # Verify using custom dataset file
    python swebench_verify_docker.py --dataset data/extended_dataset/swebench_extended.parquet
    
    # Verify 5 repositories instead of default 3
    python swebench_verify_docker.py --num-repos 5
    
    # Verify specific repositories
    python swebench_verify_docker.py --repos "psf/requests" "django/django"
    
    # Interactive mode (shows all repositories)
    python swebench_verify_docker.py --interactive
    
    # List all repositories
    python swebench_verify_docker.py --list-repos
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        help='Path to dataset file (JSON, JSONL, or Parquet). If not provided, uses original SWE-bench dataset.'
    )
    parser.add_argument(
        '--num-repos', 
        type=int, 
        default=3, 
        help='Number of repositories to verify (default: 3)'
    )
    parser.add_argument(
        '--repos', 
        nargs='+', 
        help='Specific repositories to verify (e.g., "psf/requests" "django/django")'
    )
    parser.add_argument(
        '--max-instances', 
        type=int, 
        default=2, 
        help='Maximum instances to test per repository (default: 2)'
    )
    parser.add_argument(
        '--force-rebuild', 
        action='store_true', 
        help='Force rebuild of Docker images'
    )
    parser.add_argument(
        '--list-repos',
        action='store_true',
        help='List all available repositories and exit'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactively select repositories to verify'
    )
    
    # Debug: Print all available arguments
    if len(sys.argv) > 1 and '--debug-args' in sys.argv:
        print("Available arguments:")
        parser.print_help()
        return
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # If argument parsing fails, show help and exit
        print("\nArgument parsing failed. Here's the help:")
        parser.print_help()
        sys.exit(e.code)
    
    # Handle list-repos option early
    if args.list_repos:
        print_section("Loading Dataset to List Repositories")
        try:
            # Load dataset
            if args.dataset:
                dataset, dataset_type = load_dataset_file(args.dataset)
                # Convert pandas DataFrame to list of dicts if needed
                if hasattr(dataset, 'to_dict'):
                    dataset = dataset.to_dict('records')
                print(f"✓ Loaded {len(dataset)} instances from {dataset_type.upper()} file")
            else:
                dataset = load_swebench_dataset()
                print(f"✓ Loaded {len(dataset)} instances from original SWE-bench dataset")
            
            # Get unique repositories and list them
            repo_counts = get_unique_repos(dataset)
            list_all_repositories(repo_counts)
            return
        except Exception as e:
            print(f"Error loading dataset: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print_section("SWE-bench Docker Setup Verification")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset or 'Original SWE-bench'}")
    print(f"  Number of repos: {args.num_repos}")
    print(f"  Max instances per repo: {args.max_instances}")
    print(f"  Force rebuild: {args.force_rebuild}")
    if args.repos:
        print(f"  Specific repos: {args.repos}")
    
    try:
        # Initialize Docker client
        print(f"\nInitializing Docker client...")
        client = docker.from_env()
        print(f"✓ Docker client initialized")
        
        # Load dataset
        print(f"\nLoading dataset...")
        if args.dataset:
            dataset, dataset_type = load_dataset_file(args.dataset)
            # Convert pandas DataFrame to list of dicts if needed
            if hasattr(dataset, 'to_dict'):
                dataset = dataset.to_dict('records')
            print(f"✓ Loaded {len(dataset)} instances from {dataset_type.upper()} file")
        else:
            dataset = load_swebench_dataset()
            print(f"✓ Loaded {len(dataset)} instances from original SWE-bench dataset")
        
        # Get unique repositories
        repo_counts = get_unique_repos(dataset)
        print(f"\nFound {len(repo_counts)} unique repositories in dataset")
        
        # Determine if we should use interactive mode
        use_interactive = args.interactive or (not args.repos and sys.stdin.isatty())
        
        # Show interactive selection if needed
        if use_interactive:
            list_all_repositories(repo_counts)
            selected_repos = interactive_select_repos(repo_counts)
            if not selected_repos:
                print("No repositories selected. Exiting.")
                return
        else:
            # Non-interactive mode
            if args.repos:
                selected_repos = args.repos
                # Validate that all specified repos exist in the dataset
                missing_repos = set(selected_repos) - set(repo_counts.keys())
                if missing_repos:
                    print(f"Warning: The following repositories were not found in the dataset: {missing_repos}")
                    selected_repos = [r for r in selected_repos if r not in missing_repos]
                    if not selected_repos:
                        raise ValueError("No valid repositories specified")
                # Apply num_repos limit if fewer repos are requested than specified
                if args.num_repos < len(selected_repos):
                    print(f"Limiting to {args.num_repos} repositories as specified by --num-repos")
                    selected_repos = selected_repos[:args.num_repos]
            else:
                # Select top N repositories by instance count if not already specified
                selected_repos = select_repos_for_verification(repo_counts, num_repos=args.num_repos)
        
        print(f"\nSelected {len(selected_repos)} repositories for verification:")
        for repo in selected_repos:
            print(f"  - {repo} ({repo_counts[repo]} instances)")
        
        # Filter dataset to selected repositories
        filtered_dataset = filter_dataset_by_repos(dataset, selected_repos)
        
        # Group instances by repository
        repo_instances = {}
        for instance in filtered_dataset:
            repo = instance['repo']
            if repo not in repo_instances:
                repo_instances[repo] = []
            repo_instances[repo].append(instance)
        
    except KeyboardInterrupt:
        print(f"\n\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nCritical error during verification: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
