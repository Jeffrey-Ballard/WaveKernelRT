# Offline installation

WaveKernelRT can be installed on an air-gapped system without manually
identifying its transitive Python dependencies. The repository includes a
builder that collects WaveKernelRT and every required dependency wheel into one
platform-specific bundle.

## Prerequisites

The air-gapped system must already have a compatible CPython installation with
the standard-library `venv` and `ensurepip` modules. WaveKernelRT requires
Python 3.12 or newer.

Build the bundle on an Internet-connected system matching the destination's:

- operating system;
- CPU architecture; and
- Python major and minor version.

For example, prepare a Linux x86-64 Python 3.12 bundle on Linux x86-64 with
Python 3.12. WaveKernelRT itself is platform-independent, but NumPy, SciPy,
Matplotlib, and several transitive dependencies contain compiled code. For
Linux, use the same distribution and version as the destination, or a
deliberately selected compatible build environment.

## Prepare the bundle

From a WaveKernelRT checkout on the connected preparation system, run:

```bash
python tools/build_offline_bundle.py
```

The equivalent Make target is:

```bash
make offline-bundle
```

The builder creates a descriptive directory and ZIP archive under `dist/`, for
example:

```text
dist/
|-- wavekernelrt-0-1-0-python3-12-linux-x86-64/
`-- wavekernelrt-0-1-0-python3-12-linux-x86-64.zip
```

The bundle contains:

```text
wavekernelrt-.../
|-- wheelhouse/             WaveKernelRT and all dependency wheels
|-- install.py              Standard-library-only offline installer
|-- manifest.json           Version and platform metadata plus file hashes
|-- SHA256SUMS              Transfer-verification hashes
|-- LICENSE-WaveKernelRT.txt
`-- README.txt
```

The builder requests binary wheels only. It fails instead of silently including
a source distribution that would require compilers or an Internet connection at
the destination. It also refuses to overwrite an existing output directory or
archive.

To choose another output directory or omit the ZIP archive:

```bash
python tools/build_offline_bundle.py --output /path/to/bundle
python tools/build_offline_bundle.py --output /path/to/bundle --no-archive
```

## Transfer and verify

Copy the complete directory or ZIP archive through the approved transfer path
and extract it on the air-gapped system. Do not copy individual wheels without
the manifest and installer.

Before installing, compatibility and all payload checksums can be verified:

```bash
python install.py --verify-only
```

The installer rejects a bundle prepared for a different Python minor version,
Python implementation, operating system, or CPU architecture. A successful
check does not override organization-specific software-review or media-transfer
requirements.

The included SHA-256 values detect accidental corruption or incomplete
transfers; they are not a cryptographic signature of the bundle's publisher.
Use any provenance, signing, malware-review, and third-party-license procedures
required by the destination organization.

## Install without network access

From the extracted bundle directory, run:

```bash
python install.py
```

The default destination is `.venv-wavekernelrt` in the current directory. To
select another location:

```bash
python install.py --venv /path/to/wavekernelrt-environment
```

The installer:

1. verifies the manifest, platform, and SHA-256 hashes;
2. creates a new virtual environment;
3. directs pip exclusively to the bundled `wheelhouse/` with `--no-index`;
4. checks the installed dependency set with `pip check`; and
5. imports WaveKernelRT and its primary runtime dependencies.

It refuses to overwrite an existing destination. If an installation command is
interrupted, inspect or remove the newly created destination explicitly before
trying again.

After installation, activate the environment using the path printed by the
installer. On POSIX systems this is typically:

```bash
source .venv-wavekernelrt/bin/activate
```

On Windows PowerShell:

```powershell
.venv-wavekernelrt\Scripts\Activate.ps1
```

## Install manually

Administrators may inspect the installer and use pip directly instead:

```bash
python -m venv .venv-wavekernelrt
source .venv-wavekernelrt/bin/activate
python -m pip install \
    --no-index \
    --find-links wheelhouse \
    wavekernelrt==0.1.0
python -m pip check
```

Use the version recorded in `manifest.json` rather than assuming the example
version above remains current.

## Multiple target platforms

Prepare a separate bundle for every supported combination, such as Python 3.12
on Linux x86-64 and Python 3.14 on Windows x86-64. Do not combine wheels from
different platform bundles into one directory.

For an organization with an internal Python package index, the wheelhouse can
instead be reviewed and imported into that service. Users can then follow the
organization's normal internal-index installation process.

Do not commit generated bundles or third-party wheels to the source repository.
They are platform-specific release artifacts and are already covered by the
repository's `dist/` ignore rule.
