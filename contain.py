#!/usr/bin/env python3
"""
Script to prepare Docker environment for astropy/astropy repository.
Uses the existing SWE-bench harness functions.
"""

import docker
import sys
from pathlib import Path

# Add the swebench directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from swebench.harness.docker_build import build_base_images, build_env_images
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.constants.python import MAP_REPO_VERSION_TO_SPECS_PY


def create_sample_astropy_instance():
    """Create a sample astropy instance for Docker preparation."""
    return {
        "instance_id": "astropy__astropy-sample",
        "repo": "astropy/astropy", 
        "base_commit": "abc123",  # This would be a real commit hash
        "version": "5.0",  # Use a supported version
        "problem_statement": "Sample problem for Docker preparation",
        "hints_text": "",
        "created_at": "2023-01-01T00:00:00Z",
        "test_patch": "",
        "patch": "",
        "environment_setup_commit": "abc123"
    }


def prepare_astropy_docker(force_rebuild=False):
    """Main function to prepare Docker environment for astropy."""
    print("🚀 Preparing Docker environment for astropy/astropy")
    
    # Initialize Docker client
    try:
        client = docker.from_env()
        print("✅ Docker client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Docker client: {e}")
        return False
    
    # Check if astropy is supported
    if "astropy/astropy" not in MAP_REPO_VERSION_TO_SPECS_PY:
        print("❌ astropy/astropy is not supported in the current harness")
        return False
    
    print("✅ astropy/astropy is supported")
    print(f"📋 Available versions: {list(MAP_REPO_VERSION_TO_SPECS_PY['astropy/astropy'].keys())}")
    
    # Create a sample instance
    sample_instance = create_sample_astropy_instance()
    print(f"📦 Created sample instance: {sample_instance['instance_id']}")
    
    try:
        # Create test spec for the instance
        test_spec = make_test_spec(sample_instance)
        print(f"✅ Generated test spec for version {test_spec.version}")
        astropy_spec = MAP_REPO_VERSION_TO_SPECS_PY["astropy/astropy"][test_spec.version]
        print(f"🐍 Python version: {astropy_spec['python']}")
        print(f"📦 Base image: {test_spec.base_image_key}")
        print(f"🔧 Environment image: {test_spec.env_image_key}")
        
        # Build base images
        print("\n🏗️  Building base images...")
        build_base_images(client, [sample_instance], force_rebuild=force_rebuild)
        print("✅ Base images built successfully")
        print("ℹ️  See logs in ./build/base_images/ for detailed Docker output.")

        # Build environment images  
        print("\n🏗️  Building environment images...")
        successful, failed = build_env_images(
            client, [sample_instance], force_rebuild=force_rebuild, max_workers=2
        )
        print("ℹ️  See logs in ./build/env_images/ for detailed Docker output.")
        
        if failed:
            print(f"❌ Failed to build {len(failed)} environment images")
            for failure in failed:
                print(f"   - {failure}")
            return False
        else:
            print(f"✅ Successfully built {len(successful)} environment images")
            
        print("\n🎉 Docker environment for astropy/astropy is ready!")
        print("\n📝 Next steps:")
        print("   1. You can now create instance images for specific astropy test cases")
        print("   2. Use the run_evaluation.py script to execute tests")
        print("   3. Check available images with: docker images | grep astropy")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Docker preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_astropy_specs():
    """List available astropy specifications."""
    print("\n📋 Available astropy/astropy specifications:")
    specs = MAP_REPO_VERSION_TO_SPECS_PY.get("astropy/astropy", {})
    
    for version, spec in specs.items():
        print(f"\n📦 Version {version}:")
        print(f"   🐍 Python: {spec['python']}")
        print(f"   📦 Install: {spec['install']}")
        if 'pip_packages' in spec:
            print(f"   📋 Pip packages: {len(spec['pip_packages'])} packages")
        if 'pre_install' in spec:
            print(f"   🔧 Pre-install steps: {len(spec['pre_install'])} steps")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Docker environment for astropy/astropy")
    parser.add_argument("--list", action="store_true", help="List available astropy specifications")
    parser.add_argument("--prepare", action="store_true", help="Prepare Docker environment")
    parser.add_argument("--force", action="store_true", help="Force rebuild all images (delete existing)")
    args = parser.parse_args()

    if args.list:
        list_astropy_specs()
    elif args.prepare:
        success = prepare_astropy_docker(force_rebuild=args.force)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python contain.py [--list | --prepare] [--force]")
        print("\nOptions:")
        print("  --list     List available astropy specifications")
        print("  --prepare  Prepare Docker environment for astropy")
        print("  --force    Force rebuild all images (delete existing)")
