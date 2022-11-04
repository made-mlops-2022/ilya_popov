import unittest

from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier

from ml_project.data.make_dataset import make_dataset
from ml_project.features.build_features import make_features, build_transformer, extract_target
from ml_project.train_pipeline import train_model, predict_model, make_model_pipeline
from tests.tests_params import TestTrainingPipelineParams


class TestModelsFitPredict(unittest.TestCase):
    test_params = TestTrainingPipelineParams()

    def test_train_model(self):
        data = make_dataset(self.test_params.input_data_path)

        transformer = build_transformer(self.test_params.feature_params)
        transformer.fit(data)

        target = extract_target(data, self.test_params.feature_params)
        features = make_features(transformer, data)

        model = train_model(features, target, self.test_params.train_params)
        self.assertTrue(isinstance(model, LogisticRegression))

        self.test_params.train_params.model_type = "SGDClassifier"
        model = train_model(features, target, self.test_params.train_params)
        self.assertTrue(isinstance(model, SGDClassifier))

        self.test_params.train_params.model_type = "XGBClassifier"
        model = train_model(features, target, self.test_params.train_params)
        self.assertTrue(isinstance(model, XGBClassifier))


    def test_predict_model(self):
        data = make_dataset(self.test_params.input_data_path)

        transformer = build_transformer(self.test_params.feature_params)
        transformer.fit(data)

        target = extract_target(data, self.test_params.feature_params)
        features = make_features(transformer, data)

        model = train_model(features, target, self.test_params.train_params)
        pipeline = make_model_pipeline(model, transformer)

        predict = predict_model(data, pipeline)

        self.assertEqual((100,), predict.shape)