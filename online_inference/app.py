import logging
import os
import pickle

from sklearn.pipeline import Pipeline
from fastapi import FastAPI

import pandas as pd

from entities.online_inference_params import read_online_inference_params
from entities.request_response import ConditionRequest, ConditionResponse
from download_model import download_files

DEFAULT_CONFIG = "online_inference/configs/model_path.yaml"

logger = logging.getLogger(__name__)


def load_object(path: str):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def make_pridict(data: list, feature_names: list[str], model: Pipeline) -> list:
    features = pd.DataFrame(data, columns=feature_names)
    predicts = model.predict(features)
    response = [
        ConditionResponse(condition=condition)
        for condition in predicts
    ]
    return response


app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    global model
    config_path = os.getenv("CONFIG_PATH") if os.getenv("CONFIG_PATH") else DEFAULT_CONFIG
    params = read_online_inference_params(config_path)

    if params.download_params is not None:
        download_files(params.download_params)

    model = load_object(params.model_path)


@app.get("/")
def main():
    return "Heart disease prediction service"


@app.get("/predict/", response_model=list[ConditionResponse])
def predict(request: ConditionRequest) -> list:
    return make_pridict(request.data, request.feature_names, model)


@app.get("/health")
def health():
    return not (model is None)
