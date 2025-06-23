#!/usr/bin/env python3
"""
Runner script for executing the SWE-bench task collection pipeline.

This script provides a convenient way to run the get_tasks_pipeline.py script
for collecting pull requests and converting them to task instances.
"""

import re
from pathlib import Path
from swebench.collect.get_tasks_pipeline import main as run_pipeline

# Configuration
REPOSITORIES = ["BrainLesion/panoptica"]  # List of repositories to process
OUTPUT_DIR = "data"                        # Base directory for outputs
MAX_PULLS = 20                           # No limit on number of pull requests
CUTOFF_DATE = None                         # No cutoff date for PRs


def extract_repo_info(repo_input):
    """Extract owner and repo name from GitHub URL or owner/repo format."""
    # Handle GitHub URL
    if repo_input.startswith(('http://', 'https://')):
        # Remove trailing .git if present
        repo_input = re.sub(r'\.git$', '', repo_input)
        # Extract owner/repo from URL
        match = re.search(r'github\.com[/:]([^/]+)/([^/]+)/?$', repo_input)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    # Handle owner/repo format directly
    elif '/' in repo_input:
        return repo_input
    return None


def main():
    # Create base output directory
    base_output_dir = Path(OUTPUT_DIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create repos parent directory
    repos_dir = base_output_dir / "repos"
    repos_dir.mkdir(exist_ok=True)
    
    print(f"Starting task collection for {len(REPOSITORIES)} repositories")
    
    for repo_input in REPOSITORIES:
        # Extract repo info from URL or use as is
        repo = extract_repo_info(repo_input)
        if not repo:
            print(f"\nSkipping invalid repository format: {repo_input}")
            continue
            
        # Create repo-specific directories
        repo_dir = repos_dir / repo.replace('/', '_')
        path_prs = repo_dir / "prs"
        path_tasks = repo_dir / "tasks"
        
        path_prs.mkdir(parents=True, exist_ok=True)
        path_tasks.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing repository: {repo}")
        print(f"  PR data will be saved to: {path_prs}")
        print(f"  Task data will be saved to: {path_tasks}")
        
        # Run the pipeline for this repository
        run_pipeline(
            repos=[repo],
            path_prs=str(path_prs),
            path_tasks=str(path_tasks),
            max_pulls=MAX_PULLS,
            cutoff_date=CUTOFF_DATE,
        )
    
    print("\nTask collection completed!")


if __name__ == "__main__":
    main()