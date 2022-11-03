import unittest
import numpy as np

from ml_project.data.make_dataset import make_dataset
from ml_project.features.build_features import make_features, build_transformer, extract_target
from ml_project.enities.feature_params import FeatureParams


class TestBuildFeatures(unittest.TestCase):
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
    feature_params = FeatureParams(
        numerical_features=numerical_features,
        features_to_drop=None,
        target_col="condition"
    )
    
    def test_make_features(self):
        data = make_dataset("tests/train_data_sample.csv")
        transformer = build_transformer(self.feature_params)

        transformer.fit(data)

        features = make_features(transformer, data)
        self.assertFalse(np.isnan(features).all())
        self.assertTrue(features.mean() < 1e-6)

    def test_extract_target(self):
        data = make_dataset("tests/train_data_sample.csv")

        target = extract_target(data, self.feature_params)
        self.assertEqual((50,), target.shape)
        self.assertListEqual(
            data[self.feature_params.target_col].to_list(),
            target.to_list()
        )