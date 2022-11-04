import unittest
import numpy as np

from ml_project.data.make_dataset import make_dataset
from ml_project.features.build_features import make_features, build_transformer, extract_target
from tests.tests_params import TestTrainingPipelineParams


class TestBuildFeatures(unittest.TestCase):
    test_params = TestTrainingPipelineParams()

    def test_make_features(self):
        data = make_dataset(self.test_params.input_data_path)
        transformer = build_transformer(self.test_params.feature_params)

        transformer.fit(data)

        features = make_features(transformer, data)
        self.assertFalse(np.isnan(features).all())
        self.assertTrue(features.mean() < 1e-6)

    def test_extract_target(self):
        data = make_dataset(self.test_params.input_data_path)

        target = extract_target(data, self.test_params.feature_params)
        self.assertEqual((100,), target.shape)
        self.assertListEqual(
            data[self.test_params.feature_params.target_col].to_list(),
            target.to_list()
        )