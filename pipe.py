#!/usr/bin/env python3
"""
SWE-bench Repository Selection and Docker Build Pipeline

This script combines repository selection from filter.py with Docker building from contain.py.
It allows you to:
1. Browse and select repositories from SWE-bench datasets
2. Automatically build Docker environments for selected repositories
3. Verify that the Docker setup works correctly

Usage:
    python pipe.py --dataset data/extended_dataset/swebench_extended.parquet
    python pipe.py --interactive
    python pipe.py --repos "astropy/astropy" "django/django" --force
"""

import argparse
import docker
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter

# Add the swebench directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import from filter.py functionality
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.constants import KEY_INSTANCE_ID

# Import from contain.py functionality  
from swebench.harness.docker_build import build_base_images, build_env_images
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.constants.python import MAP_REPO_VERSION_TO_SPECS_PY

def load_dataset_file(dataset_path):
    """Load dataset from various file formats"""
    if dataset_path.endswith('.json'):
        import json
        with open(dataset_path, 'r') as f:
            return json.load(f), 'json'
    elif dataset_path.endswith('.jsonl'):
        import json
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data, 'jsonl'
    elif dataset_path.endswith('.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(dataset_path)
            return df.to_dict('records'), 'parquet'
        except ImportError:
            raise ValueError("pandas is required to load Parquet files")
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")


def print_section(title: str, content: str = "", separator: str = "="):
    """Print a formatted section header"""
    print(f"\n{separator * 60}")
    print(f"{title.upper()}")
    print(f"{separator * 60}")
    if content:
        print(content)


def get_unique_repos(dataset: List[Dict]) -> Dict[str, int]:
    """Extract unique repositories from the dataset and count instances per repo."""
    repo_counts = Counter()
    for instance in dataset:
        repo = instance.get('repo', 'unknown')
        repo_counts[repo] += 1
    return dict(repo_counts)


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
    repos = sorted(repo_counts.items(), key=lambda x: x[0].lower())
    
    while True:
        print("\nEnter repository numbers or names (comma/space separated):")
        print("Examples: '1,3,5' or 'astropy/astropy django/django'")
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


def create_sample_instance_for_repo(repo: str, dataset: List[Dict]) -> Dict:
    """Create a sample instance for the given repository from the dataset."""
    # Find instances for this repo
    repo_instances = [inst for inst in dataset if inst.get('repo') == repo]
    
    if not repo_instances:
        raise ValueError(f"No instances found for repository: {repo}")
    
    # Use the first instance as a template
    sample = repo_instances[0].copy()
    
    # Modify instance_id to make it clear it's for Docker setup
    sample['instance_id'] = f"{repo.replace('/', '__')}-docker-setup"
    
    return sample


def prepare_docker_for_repo(repo: str, dataset: List[Dict], client: docker.DockerClient, force_rebuild: bool = False) -> bool:
    """Prepare Docker environment for a specific repository."""
    print(f"\n🚀 Preparing Docker environment for {repo}")
    
    try:
        # Create a sample instance for this repo
        sample_instance = create_sample_instance_for_repo(repo, dataset)
        print(f"📦 Created sample instance: {sample_instance['instance_id']}")
        
        # Create test spec for the instance
        test_spec = make_test_spec(sample_instance)
        print(f"✅ Generated test spec for version {test_spec.version}")
        print(f"📦 Base image: {test_spec.base_image_key}")
        print(f"🔧 Environment image: {test_spec.env_image_key}")
        
        # Build base images
        print(f"\n🏗️  Building base images for {repo}...")
        build_base_images(client, [sample_instance], force_rebuild=force_rebuild)
        print("✅ Base images built successfully")
        
        # Build environment images  
        print(f"\n🏗️  Building environment images for {repo}...")
        successful, failed = build_env_images(
            client, [sample_instance], force_rebuild=force_rebuild, max_workers=2
        )
        
        if failed:
            print(f"❌ Failed to build {len(failed)} environment images for {repo}")
            for failure in failed:
                print(f"   - {failure}")
            return False
        else:
            print(f"✅ Successfully built {len(successful)} environment images for {repo}")
            
        print(f"\n🎉 Docker environment for {repo} is ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error preparing Docker for {repo}: {e}")
        traceback.print_exc()
        return False


def main():
    """Main function to orchestrate the repository selection and Docker build process"""
    parser = argparse.ArgumentParser(
        description='Select SWE-bench repositories and build Docker environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode - browse and select repositories
    python pipe.py --interactive
    
    # Build Docker for specific repositories
    python pipe.py --repos "astropy/astropy" "django/django"
    
    # Use custom dataset and force rebuild
    python pipe.py --dataset data/extended_dataset/swebench_extended.parquet --force
    
    # List all available repositories
    python pipe.py --list-repos
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        help='Path to dataset file (JSON, JSONL, or Parquet). If not provided, uses original SWE-bench dataset.'
    )
    parser.add_argument(
        '--repos', 
        nargs='+', 
        help='Specific repositories to build Docker for (e.g., "astropy/astropy" "django/django")'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force rebuild of Docker images (delete existing)'
    )
    parser.add_argument(
        '--list-repos',
        action='store_true',
        help='List all available repositories and exit'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactively select repositories'
    )
    
    args = parser.parse_args()
    
    # Handle list-repos option early
    if args.list_repos:
        print_section("Loading Dataset to List Repositories")
        try:
            # Load dataset
            if args.dataset:
                dataset, dataset_type = load_dataset_file(args.dataset)
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
    
    print_section("SWE-bench Repository Selection and Docker Build Pipeline")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset or 'Original SWE-bench'}")
    print(f"  Force rebuild: {args.force}")
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
            if hasattr(dataset, 'to_dict'):
                dataset = dataset.to_dict('records')
            print(f"✓ Loaded {len(dataset)} instances from {dataset_type.upper()} file")
        else:
            dataset = load_swebench_dataset()
            print(f"✓ Loaded {len(dataset)} instances from original SWE-bench dataset")
        
        # Get unique repositories
        repo_counts = get_unique_repos(dataset)
        print(f"\nFound {len(repo_counts)} unique repositories in dataset")
        
        # Determine which repositories to build Docker for
        if args.repos:
            selected_repos = args.repos
            # Validate that all specified repos exist in the dataset
            missing_repos = set(selected_repos) - set(repo_counts.keys())
            if missing_repos:
                print(f"Warning: The following repositories were not found in the dataset: {missing_repos}")
                selected_repos = [r for r in selected_repos if r not in missing_repos]
                if not selected_repos:
                    raise ValueError("No valid repositories specified")
        elif args.interactive or sys.stdin.isatty():
            list_all_repositories(repo_counts)
            selected_repos = interactive_select_repos(repo_counts)
            if not selected_repos:
                print("No repositories selected. Exiting.")
                return
        else:
            print("No repositories specified. Use --repos, --interactive, or --list-repos")
            parser.print_help()
            return
        
        print(f"\nSelected {len(selected_repos)} repositories for Docker build:")
        for repo in selected_repos:
            print(f"  - {repo} ({repo_counts[repo]} instances)")
        
        # Build Docker environments for each selected repository
        print_section("Building Docker Environments")
        
        successful_repos = []
        failed_repos = []
        
        for i, repo in enumerate(selected_repos, 1):
            print(f"\n[{i}/{len(selected_repos)}] Processing {repo}")
            print("-" * 50)
            
            success = prepare_docker_for_repo(repo, dataset, client, force_rebuild=args.force)
            
            if success:
                successful_repos.append(repo)
                print(f"✅ {repo} completed successfully")
            else:
                failed_repos.append(repo)
                print(f"❌ {repo} failed")
        
        # Print final summary
        print_section("Build Summary")
        print(f"Repositories processed: {len(selected_repos)}")
        print(f"Successful builds: {len(successful_repos)}")
        print(f"Failed builds: {len(failed_repos)}")
        
        if successful_repos:
            print(f"\n✅ Successful repositories:")
            for repo in successful_repos:
                print(f"  - {repo}")
        
        if failed_repos:
            print(f"\n❌ Failed repositories:")
            for repo in failed_repos:
                print(f"  - {repo}")
        
        print(f"\n📝 Next steps:")
        print(f"   1. Check Docker images with: docker images | grep sweb")
        print(f"   2. Use run_evaluation.py to test specific instances")
        print(f"   3. Check build logs in ./build/ directories for details")
        
        # Exit with appropriate code
        sys.exit(0 if len(failed_repos) == 0 else 1)
        
    except KeyboardInterrupt:
        print(f"\n\nBuild process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nCritical error during build process: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
