import os
import pickle

import click
import pandas as pd
import numpy as np


def deserialize_model(input: str) -> object:
    with open(input, "rb") as f:
        model = pickle.load(f)
    return model


def predict_to_csv(target: np.ndarray, path: str) -> str:
    data = pd.Series(target)
    data.to_csv(path)
    return path


@click.command()
@click.argument("input_data_dir")
@click.argument("input_model_path")
@click.argument("output_dir")
def predict_command(input_data_dir: str, input_model_path: str, output_dir):
    data = pd.read_csv(os.path.join(input_data_dir, "data.csv"))
    if "condition" in data.columns:
        data.drop(columns="condition", inplace=True)

    model = deserialize_model(input_model_path)
    predicts = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    predict_to_csv(predicts, os.path.join(output_dir, "predictions.csv"))


if __name__ == "__main__":
    predict_command()
