import os

import pickle
import click
import pandas as pd

from sklearn.linear_model import LogisticRegression


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    penalty: str,
    max_iter: int,
    random_state: int,
) -> LogisticRegression:
    model = LogisticRegression(
        penalty=penalty, max_iter=max_iter, random_state=random_state
    )

    return model.fit(features, target)


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


@click.command()
@click.argument("input_data_dir")
@click.argument("output_model_dir")
@click.option("-p", "--penalty", default="l2", type=str)
@click.option("-m", "--max_iter", default=500, type=int)
@click.option("-r", "--random_state", default=777, type=int)
def train_command(
    input_data_dir: str,
    output_model_dir: str,
    penalty: str,
    max_iter: int,
    random_state: int,
):
    data = pd.read_csv(os.path.join(input_data_dir, "train_data.csv"))
    target = data["condition"]
    data.drop(columns="condition", inplace=True)

    model = train_model(data, target, penalty, max_iter, random_state)

    os.makedirs(output_model_dir, exist_ok=True)
    serialize_model(model, os.path.join(output_model_dir, "model.pkl"))


if __name__ == "__main__":
    train_command()
