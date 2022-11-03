import unittest

from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier

from ml_project.data.make_dataset import make_dataset
from ml_project.features.build_features import make_features, build_transformer, extract_target
from ml_project.train_pipeline import train_model, predict_model, make_model_pipeline

from ml_project.enities.training_params import TrainingParams
from ml_project.enities.feature_params import FeatureParams


class TestModelsFitPredict(unittest.TestCase):
    training_params = TrainingParams()

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

    def test_train_model(self):
        data = make_dataset("tests/train_data_sample.csv")

        transformer = build_transformer(self.feature_params)
        transformer.fit(data)

        target = extract_target(data, self.feature_params)
        features = make_features(transformer, data)

        model = train_model(features, target, self.training_params)
        self.assertTrue(isinstance(model, LogisticRegression))

        self.training_params.model_type = "SGDClassifier"
        model = train_model(features, target, self.training_params)
        self.assertTrue(isinstance(model, SGDClassifier))

        self.training_params.model_type = "XGBClassifier"
        model = train_model(features, target, self.training_params)
        self.assertTrue(isinstance(model, XGBClassifier))


    def test_predict_model(self):
        data = make_dataset("tests/train_data_sample.csv")

        transformer = build_transformer(self.feature_params)
        transformer.fit(data)

        target = extract_target(data, self.feature_params)
        features = make_features(transformer, data)

        model = train_model(features, target, self.training_params)
        pipeline = make_model_pipeline(model, transformer)

        predict = predict_model(data, pipeline)

        self.assertEqual((50,), predict.shape)