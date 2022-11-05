import unittest
import os

import numpy as np

from tempfile import TemporaryDirectory

from ml_project.data.make_dataset import (
    make_dataset,
    split_train_test_data,
    predict_to_csv,
)
from tests.tests_params import TestTrainingPipelineParams


class TestMakeDataset(unittest.TestCase):
    test_params = TestTrainingPipelineParams()

    def test_make_dataset(self):
        data = make_dataset(self.test_params.input_data_path)

        self.assertEqual(100, data.shape[0])
        self.assertEqual(14, data.shape[1])

        expected_columns = [
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
            "condition",
        ]
        self.assertListEqual(expected_columns, list(data.columns))

    def test_split_data(self):
        data = make_dataset(self.test_params.input_data_path)

        test_size = self.test_params.splitting_params.test_size

        train, test = split_train_test_data(data, self.test_params.splitting_params)
        expected_train_size = np.round(data.shape[0] * (1 - test_size))
        expected_test_size = np.round(data.shape[0] * test_size)

        self.assertTrue(train.shape[0] == expected_train_size)
        self.assertTrue(test.shape[0] == expected_test_size)

    def test_predict_to_csv(self):
        predict = np.random.randint(2, size=10)

        with TemporaryDirectory() as tempdir:
            path = f"{tempdir}/test_predict.csv"
            predict_to_csv(predict, path)
            self.assertTrue(os.path.exists(path))
