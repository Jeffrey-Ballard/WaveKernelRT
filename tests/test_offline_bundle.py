import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0,str(REPOSITORY_ROOT))

from tools import build_offline_bundle,install_offline_bundle


class TestOfflineBundle(unittest.TestCase):
    def test_safe_name_is_suitable_for_bundle_paths(self):
        self.assertEqual(build_offline_bundle.safe_name("macOS / ARM64"),"macos-arm64")

    def test_verify_files_detects_modified_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp).resolve()
            payload = bundle / "payload.whl"
            payload.write_bytes(b"original wheel bytes")
            manifest = {
                "files": [
                    {
                        "path": payload.name,
                        "size": payload.stat().st_size,
                        "sha256": install_offline_bundle.sha256(payload),
                    }
                ]
            }

            with patch.object(install_offline_bundle,"BUNDLE_ROOT",bundle):
                install_offline_bundle.verify_files(manifest)
                payload.write_bytes(b"modified wheel bytes")
                with self.assertRaisesRegex(RuntimeError,"wrong size|Checksum"):
                    install_offline_bundle.verify_files(manifest)

    def test_manifest_path_cannot_leave_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp).resolve()
            with patch.object(install_offline_bundle,"BUNDLE_ROOT",bundle):
                with self.assertRaisesRegex(RuntimeError,"Unsafe path"):
                    install_offline_bundle.bundle_file("../outside.whl")

    def test_incompatible_python_version_is_rejected(self):
        manifest = {
            "python": {
                "major_minor": "0.0",
                "implementation": platform.python_implementation(),
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
        }
        expected = f"running Python {sys.version_info.major}.{sys.version_info.minor}"
        with self.assertRaisesRegex(RuntimeError,expected):
            install_offline_bundle.verify_compatibility(manifest)


if __name__ == "__main__":
    unittest.main()
