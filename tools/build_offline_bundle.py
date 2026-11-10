"""Build a platform-specific, ready-to-transfer WaveKernelRT wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_SOURCE = Path(__file__).with_name("install_offline_bundle.py")


def safe_name(value: str) -> str:
    """Convert metadata or platform text into a filename component."""
    characters: list[str] = []
    for character in value:
        if character.isalnum():
            characters.append(character.lower())
        elif characters and characters[-1] != "-":
            characters.append("-")
    return "".join(characters).strip("-")


def project_metadata() -> tuple[str, str]:
    """Read the package name and version without importing the project."""
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    return project["name"], project["version"]


def bundle_name(package: str, version: str) -> str:
    """Return a descriptive name for a bundle built on this host."""
    python = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return "-".join(
        safe_name(value)
        for value in (
            package,
            version,
            python,
            platform.system(),
            platform.machine(),
        )
    )


def run(command: list[str], *, cwd: Path) -> None:
    """Run one bundle-construction command."""
    environment = os.environ.copy()
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, bundle_root: Path) -> dict[str, Any]:
    """Create one portable manifest record."""
    return {
        "path": path.relative_to(bundle_root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_readme(
    path: Path,
    *,
    package: str,
    version: str,
    name: str,
) -> None:
    """Write concise instructions inside the generated bundle."""
    path.write_text(
        f"""WaveKernelRT offline installation bundle
========================================

Package: {package} {version}
Bundle:  {name}
Python:  {platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}
System:  {platform.system()} {platform.machine()}

This bundle contains WaveKernelRT and all Python dependency wheels needed for
the platform above. Copy or extract the complete directory onto the matching
air-gapped system, then run:

    python install.py

To choose the virtual-environment location:

    python install.py --venv /path/to/environment

To verify compatibility and file checksums without installing:

    python install.py --verify-only

The installer uses pip with --no-index and cannot contact a package index.
Python must include the standard-library venv and ensurepip modules. See the
repository's offline-installation documentation for platform compatibility,
administrative deployment, and troubleshooting guidance.
""",
        encoding="utf-8",
    )


def build_bundle(bundle_root: Path, package: str, version: str, name: str) -> None:
    """Populate one bundle directory."""
    wheelhouse = bundle_root / "wheelhouse"
    build_directory = bundle_root.parent / "project-wheel"
    wheelhouse.mkdir(parents=True)
    build_directory.mkdir()

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(REPOSITORY_ROOT),
            "--no-deps",
            "--wheel-dir",
            str(build_directory),
        ],
        cwd=bundle_root.parent,
    )

    project_wheels = list(build_directory.glob("*.whl"))
    if len(project_wheels) != 1:
        raise RuntimeError(
            f"Expected one project wheel, found {len(project_wheels)} in "
            f"{build_directory}"
        )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            str(project_wheels[0]),
            "--dest",
            str(wheelhouse),
            "--only-binary=:all:",
        ],
        cwd=bundle_root.parent,
    )

    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise RuntimeError("pip did not place any wheels in the bundle")
    if any(path.suffix != ".whl" for path in wheelhouse.iterdir()):
        raise RuntimeError("The wheelhouse contains a non-wheel distribution")

    installer = bundle_root / "install.py"
    readme = bundle_root / "README.txt"
    license_file = bundle_root / "LICENSE-WaveKernelRT.txt"
    shutil.copy2(INSTALLER_SOURCE, installer)
    shutil.copy2(REPOSITORY_ROOT / "LICENSE", license_file)
    write_readme(readme, package=package, version=version, name=name)

    payload = [installer, readme, license_file, *wheels]
    records = [file_record(path, bundle_root) for path in payload]
    manifest = {
        "schema_version": 1,
        "package": package,
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "files": records,
    }
    (bundle_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle_root / "SHA256SUMS").write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a WaveKernelRT wheelhouse for the current Python, operating "
            "system, and architecture."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: dist/<generated bundle name>)",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="do not create a ZIP archive beside the bundle directory",
    )
    return parser.parse_args()


def main() -> None:
    """Build an offline bundle without overwriting existing output."""
    args = parse_args()
    package, version = project_metadata()
    name = bundle_name(package, version)
    output = (args.output or REPOSITORY_ROOT / "dist" / name).expanduser().resolve()
    archive = Path(f"{output}.zip")

    if output.exists():
        raise RuntimeError(f"Output directory already exists: {output}")
    if not args.no_archive and archive.exists():
        raise RuntimeError(f"Output archive already exists: {archive}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wavekernelrt-offline-build-", dir=output.parent
    ) as temp:
        temporary_root = Path(temp)
        temporary_bundle = temporary_root / output.name
        temporary_bundle.mkdir()
        build_bundle(temporary_bundle, package, version, name)
        temporary_bundle.replace(output)

    print(f"Offline bundle created: {output}")
    if not args.no_archive:
        archive_path = shutil.make_archive(
            str(output),
            "zip",
            root_dir=output.parent,
            base_dir=output.name,
        )
        print(f"Transfer archive created: {archive_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Offline bundle build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
