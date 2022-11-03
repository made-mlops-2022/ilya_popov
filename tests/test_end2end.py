import unittest
import os

from tempfile import TemporaryDirectory

from ml_project.enities.train_pipeline_params import TrainingPipelineParams
from ml_project.enities.split_params import SplittingParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.training_params import TrainingParams

from ml_project.train_pipeline import run_train_pipeline


class TestEnd2End(unittest.TestCase):
    numerical_features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    training_pipeline_params = TrainingPipelineParams(
        input_data_path="tests/train_data_sample.csv",
        output_model_path="",
        metric_path="",
        splitting_params=SplittingParams(test_size=0.3, random_state=40),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            features_to_drop=None,
            target_col="condition"
        ),
        train_params=TrainingParams(random_state=40),
    )
    
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
