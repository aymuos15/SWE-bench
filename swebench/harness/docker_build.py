from __future__ import annotations

import docker
import docker.errors
import logging
import sys
import time
import traceback

from pathlib import Path

from swebench.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    DOCKER_USER,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    UTF8,
)
from swebench.harness.docker_utils import cleanup_container, remove_image
from swebench.harness.test_spec.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec,
)
from swebench.harness.utils import ansi_escape, run_threadpool


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(instance_id: str, log_file: Path, mode="w", add_stdout: bool = False):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.

    If `add_stdout` is True, logs will also be sent to stdout, which can be used for
    streaming ephemeral output from Modal containers.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    if add_stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"%(asctime)s - {instance_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client: docker.DockerClient,
    build_dir: Path,
    nocache: bool = False,
):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log")
    start_time = time.time()
    
    logger.info(f"{'='*50}")
    logger.info(f"STARTING BUILD: {image_name}")
    logger.info(f"{'='*50}")
    logger.info(f"Build directory: {build_dir}")
    logger.info(f"Platform: {platform}")
    logger.info(f"Using cache: {not nocache}")
    logger.info(f"Number of setup scripts: {len(setup_scripts)}")
    
    print(f"\n[BUILD] Starting build for {image_name}")
    
    try:
        # Write the setup scripts to the build directory
        logger.info("Writing setup scripts...")
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(f"Setup script {setup_script_name} may not be used in Dockerfile")
        
        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)
        
        # Log Dockerfile for reference
        logger.info(f"Dockerfile contents:\n{dockerfile}")
        
        # Build the image
        logger.info(f"Starting Docker build for {image_name}")
        print(f"[BUILD] Building {image_name}...")
        
        build_start = time.time()
        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
        )
        
        # Track build steps and progress
        step_count = 0
        current_step = 0
        buildlog = ""
        
        # First pass to count total steps
        dockerfile_lines = dockerfile.split('\n')
        step_count = sum(1 for line in dockerfile_lines if line.strip().startswith(('RUN', 'COPY', 'ADD', 'CMD', 'ENTRYPOINT')))
        
        logger.info(f"Build will execute approximately {step_count} steps")
        
        # Process build output
        for chunk in response:
            if "stream" in chunk:
                chunk_stream = ansi_escape(chunk["stream"]).strip()
                if chunk_stream:
                    # Check for step completion
                    if chunk_stream.startswith("Step "):
                        current_step += 1
                        progress = f"[{current_step}/{step_count}]" if step_count > 0 else ""
                        print(f"[BUILD] {progress} {chunk_stream}")
                    
                    # Log the output
                    logger.info(chunk_stream)
                    buildlog += chunk_stream + "\n"
                    
                    # Show progress for long-running steps
                    if any(x in chunk_stream.lower() for x in ["installing", "downloading", "building", "compiling"]):
                        print(f"[BUILD] {chunk_stream}")
                        
            elif "status" in chunk:
                status = chunk["status"].strip()
                if "progress" in chunk and "id" in chunk:
                    # Show progress for download/extraction steps
                    progress = chunk.get("progressDetail", {})
                    if "current" in progress and "total" in progress and progress["total"] > 0:
                        pct = (progress["current"] / progress["total"]) * 100
                        print(f"[BUILD] {status}: {pct:.1f}%", end="\r")
            
            elif "errorDetail" in chunk:
                error_msg = ansi_escape(chunk["errorDetail"].get("message", "Unknown error"))
                logger.error(f"Build error: {error_msg}")
                print(f"[ERROR] Build failed: {error_msg}")
                raise docker.errors.BuildError(error_msg, buildlog)
        
        build_time = time.time() - build_start
        logger.info(f"Build completed successfully in {build_time:.2f} seconds")
        print(f"[BUILD] ✓ Successfully built {image_name} in {build_time:.1f}s")
        
    except docker.errors.BuildError as e:
        error_msg = str(e)
        logger.error(f"Build failed: {error_msg}")
        print(f"[ERROR] Build failed: {error_msg}")
        raise BuildImageError(image_name, error_msg, logger) from e
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error: {error_msg}")
        print(f"[ERROR] Unexpected error: {error_msg}")
        raise BuildImageError(image_name, error_msg, logger) from e
        
    finally:
        total_time = time.time() - start_time
        logger.info(f"Total build time: {total_time:.2f} seconds")
        close_logger(logger)  # functions that create loggers should close them


def build_base_images(
    client: docker.DockerClient, dataset: list, force_rebuild: bool = False
):
    """
    Builds the base images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    # Get the base images to build from the dataset
    test_specs = get_test_specs_from_dataset(dataset)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }

    # Build the base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            # Check if the base image already exists
            client.images.get(image_name)
            if force_rebuild:
                # Remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        except docker.errors.ImageNotFound:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")


def get_env_configs_to_build(
    client: docker.DockerClient,
    dataset: list,
):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    test_specs = get_test_specs_from_dataset(dataset)

    for test_spec in test_specs:
        # Check if the base image exists
        try:
            if test_spec.base_image_key not in base_images:
                base_images[test_spec.base_image_key] = client.images.get(
                    test_spec.base_image_key
                )
            base_image = base_images[test_spec.base_image_key]
        except docker.errors.ImageNotFound:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                "Please build the base images first."
            )

        # Check if the environment image exists
        image_exists = False
        try:
            env_image = client.images.get(test_spec.env_image_key)
            image_exists = True
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts


def build_env_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
):
    """
    Builds the environment images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    print("\n" + "="*50)
    print("BUILDING ENVIRONMENT IMAGES")
    print("="*50)
    
    # Get the environment images to build from the dataset
    if force_rebuild:
        env_image_keys = {x.env_image_key for x in get_test_specs_from_dataset(dataset)}
        print(f"Force rebuild enabled - removing {len(env_image_keys)} existing images")
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    
    # Build base images first
    print("\nBuilding base images...")
    build_base_images(client, dataset, force_rebuild)
    
    # Get configs for environment images that need to be built
    configs_to_build = get_env_configs_to_build(client, dataset)
    if len(configs_to_build) == 0:
        print("\n✓ No environment images need to be built.")
        return [], []
        
    print(f"\nPreparing to build {len(configs_to_build)} environment images...")
    
    # Prepare build arguments
    args_list = []
    for image_name, config in configs_to_build.items():
        build_dir = ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__")
        build_dir.mkdir(parents=True, exist_ok=True)
        
        args_list.append(
            (
                image_name,
                {"setup_env.sh": config["setup_script"]},
                config["dockerfile"],
                config["platform"],
                client,
                build_dir,
            )
        )
    
    # Track build progress
    print(f"\nStarting builds with {max_workers} parallel workers...")
    start_time = time.time()
    
    # Run builds in parallel
    successful, failed = run_threadpool(build_image, args_list, max_workers)
    
    # Calculate and display build statistics
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("BUILD SUMMARY")
    print("="*50)
    
    if successful:
        print(f"✓ Successfully built {len(successful)} images")
    if failed:
        print(f"✗ Failed to build {len(failed)} images")
    
    print(f"\nTotal build time: {total_time:.2f} seconds")
    
    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    tag: str = None,
):
    """
    Builds the instance images required for the dataset if they do not already exist.

    Args:
        dataset (list): List of test specs or dataset to build images for
        client (docker.DockerClient): Docker client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Build environment images (and base images as needed) first
    test_specs = list(
        map(
            lambda x: make_test_spec(x, namespace=namespace, instance_image_tag=tag),
            dataset,
        )
    )
    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")
    _, env_failed = build_env_images(client, test_specs, force_rebuild, max_workers)

    if len(env_failed) > 0:
        # Don't build images for instances that depend on failed-to-build env images
        dont_run_specs = [
            spec for spec in test_specs if spec.env_image_key in env_failed
        ]
        test_specs = [
            spec for spec in test_specs if spec.env_image_key not in env_failed
        ]
        print(
            f"Skipping {len(dont_run_specs)} instances - due to failed env image builds"
        )
    print(f"Building instance images for {len(test_specs)} instances")
    successful, failed = list(), list()

    # `logger` is set to None b/c logger is created in build-instage_image
    payloads = [(spec, client, None, False) for spec in test_specs]
    # Build the instance images
    successful, failed = run_threadpool(build_instance_image, payloads, max_workers)
    # Show how many images failed to build
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_image(
    test_spec: TestSpec,
    client: docker.DockerClient,
    logger: logging.Logger | None,
    nocache: bool,
):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """
    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
        ":", "__"
    )
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key
    dockerfile = test_spec.instance_dockerfile

    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(env_image_name)
    except docker.errors.ImageNotFound as e:
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment image {env_image_name} not found for {test_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name} for {test_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        client.images.get(image_name)
        image_exists = True
    except docker.errors.ImageNotFound:
        pass

    # Build the instance image
    if not image_exists:
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": test_spec.install_repo_script,
            },
            dockerfile=dockerfile,
            platform=test_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=nocache,
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)


def build_container(
    test_spec: TestSpec,
    client: docker.DockerClient,
    run_id: str,
    logger: logging.Logger,
    nocache: bool,
    force_rebuild: bool = False,
):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    if not test_spec.is_remote_image:
        build_instance_image(test_spec, client, logger, nocache)
    else:
        try:
            client.images.get(test_spec.instance_image_key)
        except docker.errors.ImageNotFound:
            try:
                client.images.pull(test_spec.instance_image_key)
            except docker.errors.NotFound as e:
                raise BuildImageError(test_spec.instance_id, str(e), logger) from e
            except Exception as e:
                raise Exception(
                    f"Error occurred while pulling image {test_spec.base_image_key}: {str(e)}"
                )

    container = None
    try:
        # Create the container
        logger.info(f"Creating container for {test_spec.instance_id}...")

        # Define arguments for running the container
        run_args = test_spec.docker_specs.get("run_args", {})
        cap_add = run_args.get("cap_add", [])

        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=cap_add,
        )
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e
