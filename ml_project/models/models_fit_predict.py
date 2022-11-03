import pandas as pd
import pickle

from typing import Dict, Union

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml_project.enities.training_params import TrainingParams


ClassifierModel = Union[LogisticRegression, SGDClassifier, XGBClassifier]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    training_params: TrainingParams
) -> ClassifierModel:
    match training_params.model_type:
        case "LogisticRegression":
            model = LogisticRegression(random_state=training_params.random_state, n_jobs=training_params.n_jobs)
        
        case "SGDClassifier":
            model = SGDClassifier(random_state=training_params.random_state, n_jobs=training_params.n_jobs)

        case "XGBClassifier":
            model = XGBClassifier()

    return model.fit(features, target)


def predict_model(features: pd.DataFrame, model: Pipeline) -> pd.Series:
    return model.predict(features)


def evaluate_metrics(true: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    return {
        "F1": f1_score(true, predicted),
        "RocAuc": roc_auc_score(true, predicted),
        "NLLLoss": log_loss(true, predicted)
    }


def make_model_pipeline(model: ClassifierModel, transformer: ColumnTransformer) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def deserialize_model(input: str) -> object:
    with open(input, "rb") as f:
        model = pickle.load(f)
    return model
    