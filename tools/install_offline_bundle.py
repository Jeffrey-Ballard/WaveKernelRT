"""Install WaveKernelRT from a prepared offline bundle.

This file is copied into each bundle as ``install.py``. It intentionally uses
only the Python standard library so it can run before third-party packages are
installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parent


def normalized_platform_value(value: str) -> str:
    """Normalize a platform or architecture name for comparison."""
    return value.strip().lower().replace("-", "_")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    """Load the bundle manifest."""
    manifest_path = BUNDLE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Bundle manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_compatibility(manifest: dict[str, Any]) -> None:
    """Reject a bundle prepared for a different Python or platform."""
    expected_python = manifest["python"]["major_minor"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise RuntimeError(
            f"Bundle requires Python {expected_python}; running Python {actual_python}"
        )

    expected_implementation = normalized_platform_value(
        manifest["python"]["implementation"]
    )
    actual_implementation = normalized_platform_value(platform.python_implementation())
    if actual_implementation != expected_implementation:
        raise RuntimeError(
            "Bundle requires Python implementation "
            f"{manifest['python']['implementation']}; running "
            f"{platform.python_implementation()}"
        )

    expected_system = normalized_platform_value(manifest["platform"]["system"])
    actual_system = normalized_platform_value(platform.system())
    if actual_system != expected_system:
        raise RuntimeError(
            f"Bundle requires {manifest['platform']['system']}; running "
            f"{platform.system()}"
        )

    expected_machine = normalized_platform_value(manifest["platform"]["machine"])
    actual_machine = normalized_platform_value(platform.machine())
    if actual_machine != expected_machine:
        raise RuntimeError(
            f"Bundle requires architecture {manifest['platform']['machine']}; "
            f"running {platform.machine()}"
        )


def bundle_file(relative_name: str) -> Path:
    """Resolve a manifest path while preventing traversal outside the bundle."""
    relative_path = Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Unsafe path in bundle manifest: {relative_name!r}")

    resolved = (BUNDLE_ROOT / relative_path).resolve()
    if not resolved.is_relative_to(BUNDLE_ROOT):
        raise RuntimeError(f"Path leaves bundle directory: {relative_name!r}")
    return resolved


def verify_files(manifest: dict[str, Any]) -> None:
    """Verify every payload file recorded in the manifest."""
    for record in manifest["files"]:
        path = bundle_file(record["path"])
        if not path.is_file():
            raise RuntimeError(f"Bundle file is missing: {record['path']}")

        actual_size = path.stat().st_size
        if actual_size != record["size"]:
            raise RuntimeError(
                f"Bundle file has the wrong size: {record['path']} "
                f"({actual_size} != {record['size']})"
            )

        actual_hash = sha256(path)
        if actual_hash != record["sha256"]:
            raise RuntimeError(f"Checksum failed for bundle file: {record['path']}")


def environment_python(environment: Path) -> Path:
    """Return the virtual environment's Python executable."""
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    """Run one offline installation command."""
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def install(manifest: dict[str, Any], destination: Path) -> None:
    """Create a virtual environment and install the bundled distribution."""
    if destination.exists():
        raise RuntimeError(
            f"Installation destination already exists: {destination}. "
            "Choose another path or remove it explicitly."
        )

    wheelhouse = BUNDLE_ROOT / "wheelhouse"
    if not wheelhouse.is_dir():
        raise RuntimeError(f"Bundle wheelhouse is missing: {wheelhouse}")

    print(f"Creating virtual environment: {destination}", flush=True)
    try:
        venv.EnvBuilder(with_pip=True).create(destination)
    except Exception as exc:
        raise RuntimeError(
            "Could not create a virtual environment. Confirm that this Python "
            "installation includes the venv and ensurepip modules."
        ) from exc

    python = environment_python(destination)
    pip_environment = os.environ.copy()
    pip_environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }
    )

    package = manifest["package"]
    version = manifest["version"]
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            f"{package}=={version}",
        ],
        cwd=BUNDLE_ROOT,
        environment=pip_environment,
    )
    run(
        [str(python), "-m", "pip", "check"],
        cwd=BUNDLE_ROOT,
        environment=pip_environment,
    )

    smoke_test = """
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy
import scipy
import wavekernelrt
from wavekernelrt import Geometry

print("WaveKernelRT version:", version("wavekernelrt"))
print("Package location:", Path(wavekernelrt.__file__).resolve())
print("Offline installation imports passed.")
"""
    run(
        [str(python), "-I", "-c", smoke_test],
        cwd=BUNDLE_ROOT,
        environment=pip_environment,
    )

    if sys.platform == "win32":
        activation = destination / "Scripts" / "Activate.ps1"
    else:
        activation = destination / "bin" / "activate"
    print(f"\nInstallation complete. Activate it with:\n  {activation}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Verify and install a prepared WaveKernelRT offline bundle."
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=Path.cwd() / ".venv-wavekernelrt",
        help="virtual environment to create (default: ./.venv-wavekernelrt)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify compatibility and checksums without installing",
    )
    return parser.parse_args()


def main() -> None:
    """Verify the bundle and optionally install it."""
    args = parse_args()
    manifest = load_manifest()
    verify_compatibility(manifest)
    verify_files(manifest)
    print("Bundle compatibility and checksums passed.", flush=True)

    if not args.verify_only:
        install(manifest, args.venv.expanduser().resolve())


if __name__ == "__main__":
    try:
        main()
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Offline installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
