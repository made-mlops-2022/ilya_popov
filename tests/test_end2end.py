import unittest
import os

from tempfile import TemporaryDirectory

from tests.tests_params import TestTrainingPipelineParams

from ml_project.train_pipeline import run_train_pipeline


class TestEnd2End(unittest.TestCase):
    training_pipeline_params = TestTrainingPipelineParams()
    
    def test_end2end(self):
        with TemporaryDirectory() as tempdir:
            expected_model_path = f"{tempdir}/model.pkl"
            self.training_pipeline_params.output_model_path = expected_model_path

            expected_metric_path = f"{tempdir}/metrics.json"
            self.training_pipeline_params.metric_path = expected_metric_path

            path_to_model, metrics = run_train_pipeline(self.training_pipeline_params)

            self.assertTrue(os.path.exists(path_to_model))
            self.assertTrue(metrics["F1"] > 0)
            self.assertTrue(metrics["RocAuc"] != 0.5)
