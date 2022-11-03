import unittest
import os

import numpy as np

from tempfile import TemporaryDirectory

from ml_project.data.make_dataset import make_dataset, split_train_test_data, predict_to_csv
from ml_project.enities.split_params import SplittingParams


class TestMakeDataset(unittest.TestCase):
    def test_make_dataset(self):
        data = make_dataset("tests/train_data_sample.csv")

        self.assertEqual(50, data.shape[0])
        self.assertEqual(14, data.shape[1])

        expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']
        self.assertListEqual(expected_columns, list(data.columns))

    def test_split_data(self):
        data = make_dataset("tests/train_data_sample.csv")
        
        test_size = 0.2
        splitting_params = SplittingParams(random_state=1, test_size=test_size, shuffle=False)

        train, test = split_train_test_data(data, splitting_params)
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

        