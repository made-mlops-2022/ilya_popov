from copy import copy
from unittest import TestCase
from unittest.mock import patch
from fastapi.testclient import TestClient

import numpy as np

from app import app


def create_dataset(items_count):
    dataset = [
        list(20 * np.random.randn(13) + 100) for _ in range(items_count)
    ]
    return dataset


class TestApp(TestCase):
    client = TestClient(app)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    def test_health(self):
        with patch("app.model"):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json())

    def test_health_false(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json())

    def test_predict(self):
        with patch("app.model") as model_mock:
            model_mock.predict.return_value = np.random.randint(2, size=10)
            dataset = create_dataset(10)

            response = self.client.get(
                "/predict/",
                json={"data": dataset, "feature_names": self.feature_names}
            )
            self.assertEqual(response.status_code, 200)

    def test_predict_invalid(self):
        with patch("app.model") as model_mock:
            model_mock.predict.return_value = np.random.randint(2, size=10)
            dataset = create_dataset(10)

            response = self.client.get(
                "/predict/",
                json={"data": dataset, "feature_names": ["akdjf", 123]}
            )
            self.assertGreaterEqual(response.status_code, 400)

            invalid_feature_names = copy(self.feature_names)
            invalid_feature_names[:] = ["qwerty"] * 13
            print(invalid_feature_names)
            response = self.client.get(
                "/predict/",
                json={"data": dataset, "feature_names": []}
            )
            self.assertGreaterEqual(response.status_code, 400)

            response = self.client.get(
                "/predict/",
                json={"data": dataset[0], "feature_names": self.feature_names}
            )
            self.assertGreaterEqual(response.status_code, 400)

            del dataset[0][0]
            response = self.client.get(
                "/predict/",
                json={"data": dataset, "feature_names": self.feature_names}
            )
            self.assertGreaterEqual(response.status_code, 400)
