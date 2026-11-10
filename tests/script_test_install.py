"""Build an offline bundle and test its isolated installation path."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str],*,cwd: Path) -> None:
    """Run one installation-test command and fail immediately on an error."""
    subprocess.run(command,cwd=cwd,check=True)


def environment_python(environment: Path) -> Path:
    """Return the Python executable created by the offline installer."""
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    """Build, offline-install, and import-test the project distribution."""
    with tempfile.TemporaryDirectory(prefix="wavekernelrt-install-test-") as temp:
        work_directory = Path(temp)
        bundle = work_directory / "bundle"
        environment = work_directory / "venv"

        run([sys.executable,str(REPOSITORY_ROOT / "tools" / "build_offline_bundle.py"),"--output",str(bundle),"--no-archive"],cwd=REPOSITORY_ROOT)

        run([sys.executable,str(bundle / "install.py"),"--venv",str(environment)],cwd=work_directory)

        python = environment_python(environment)
        smoke_test = f"""
from pathlib import Path

import wavekernelrt

package_path = Path(wavekernelrt.__file__).resolve()
repository_root = Path({str(REPOSITORY_ROOT)!r}).resolve()
if package_path.is_relative_to(repository_root):
    raise RuntimeError(f"Imported WaveKernelRT from source checkout: {{package_path}}")

print("Interpreter:", {str(python)!r})
print("Package location:", package_path)
print("Offline-bundle installation passed.")
"""
        run([str(python),"-I","-c",smoke_test],cwd=work_directory)


if __name__ == "__main__":
    main()
