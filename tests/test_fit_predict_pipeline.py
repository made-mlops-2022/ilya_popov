import unittest
import os

from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.tests_params import TestPredictPipelineParams, TestTrainingPipelineParams

from ml_project.train_pipeline import run_train_pipeline
from ml_project.predict_pipeline import run_predict_pipeline


class TestFitPredictPipeline(unittest.TestCase):
    training_pipeline_params = TestTrainingPipelineParams()
    predict_pipeline_params = TestPredictPipelineParams()

    def test_end2end_train(self):
        with TemporaryDirectory() as tempdir:
            expected_model_path = f"{tempdir}/model.pkl"
            self.training_pipeline_params.output_model_path = expected_model_path

            expected_metric_path = f"{tempdir}/metrics.json"
            self.training_pipeline_params.metric_path = expected_metric_path

            with patch("ml_project.train_pipeline.logger"):
                path_to_model, metrics = run_train_pipeline(self.training_pipeline_params)

            self.assertTrue(os.path.exists(path_to_model))
            self.assertTrue(metrics["F1"] > 0)
            self.assertTrue(metrics["RocAuc"] != 0.5)

    def test_end2end_predict(self):
        self.assertTrue(os.path.exists(self.predict_pipeline_params.input_model_path))
        with TemporaryDirectory() as tempdir:
            expected_predict_path = f"{tempdir}/predict.csv"
            self.predict_pipeline_params.output_data_path = expected_predict_path

            with patch("ml_project.predict_pipeline.logger"):
                real_predict_path = run_predict_pipeline(self.predict_pipeline_params)

            self.assertTrue(os.path.exists(real_predict_path))
