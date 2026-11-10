import subprocess
import sys
import unittest

import wavekernelrt
import wavekernelrt.pipeline as pipeline_api
from wavekernelrt.core.beampattern import BeamPattern
from wavekernelrt.core.geometry import Geometry
from wavekernelrt.core.shading import Shading
from wavekernelrt.core.steeringvectors import SteeringVectors
from wavekernelrt.core.steeringvectors_utils import construct_rbe_X3
from wavekernelrt.pipeline.build_pipeline import build_pipeline
from wavekernelrt.pipeline.configs import PipelineConfig
from wavekernelrt.pipeline.run_pipeline import run_pipeline


class TestPublicAPI(unittest.TestCase):
    def test_plain_package_import_is_lazy(self):
        result = subprocess.run([sys.executable,"-c",("import sys; import wavekernelrt; "
            "assert 'matplotlib' not in sys.modules; " "assert 'scipy' not in sys.modules")],
            check=False,capture_output=True,text=True)

        self.assertEqual(result.returncode,0,result.stderr)

    def test_top_level_exports_are_implementation_aliases(self):
        self.assertIs(wavekernelrt.Geometry,Geometry)
        self.assertIs(wavekernelrt.Shading,Shading)
        self.assertIs(wavekernelrt.SteeringVectors,SteeringVectors)
        self.assertIs(wavekernelrt.BeamPattern,BeamPattern)
        self.assertIs(wavekernelrt.construct_rbe_X3,construct_rbe_X3)
        self.assertIs(wavekernelrt.PipelineConfig,PipelineConfig)
        self.assertIs(wavekernelrt.build_pipeline,build_pipeline)
        self.assertIs(wavekernelrt.run_pipeline,run_pipeline)

    def test_top_level_all_names_resolve(self):
        for name in wavekernelrt.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(wavekernelrt,name))

    def test_pipeline_namespace_exports_resolve(self):
        for name in pipeline_api.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(pipeline_api,name))


if __name__ == "__main__":
    unittest.main()
