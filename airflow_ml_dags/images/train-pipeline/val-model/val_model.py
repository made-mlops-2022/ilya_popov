import json
import os
import pickle

from typing import Dict

import pandas as pd
import click

from sklearn.metrics import f1_score, roc_auc_score, log_loss


def deserialize_model(input: str) -> object:
    with open(input, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate_metrics(true: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    return {
        "F1": f1_score(true, predicted),
        "RocAuc": roc_auc_score(true, predicted),
        "NLLLoss": log_loss(true, predicted),
    }


@click.command()
@click.argument("input_data_path")
@click.argument("input_model_path")
def val_model_command(input_data_path: str, input_model_path: str):
    data = pd.read_csv(os.path.join(input_data_path, "test_data.csv"))
    y_test = data["condition"]
    data.drop(columns="condition", inplace=True)

    model = deserialize_model(os.path.join(input_model_path, "model.pkl"))

    predicts = model.predict(data)
    metrics = evaluate_metrics(y_test, predicts)

    with open(os.path.join(input_model_path, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == "__main__":
    val_model_command()
