import os

from typing import List

import click
import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


FEATURES_LIST = [
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


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def build_transformer(features: List[str]) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                features,
            ),
        ]
    )
    return transformer


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scale", StandardScaler()),
        ]
    )
    return num_pipeline


def extract_target(data: pd.DataFrame, target: str) -> pd.Series:
    return data[target]


@click.command()
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("-f", "--features", default=FEATURES_LIST, type=list)
def preprocess_command(
    input_dir: str, output_dir: str, features: List[str] = FEATURES_LIST
):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    transformer = build_transformer(features)
    transformer.fit(data)

    processed_data = make_features(transformer, data)

    os.makedirs(output_dir, exist_ok=True)
    processed_data = pd.DataFrame(processed_data, columns=features)
    processed_data["condition"] = data["condition"]
    processed_data.to_csv(os.path.join(output_dir, "data.csv"), index=False)


if __name__ == "__main__":
    preprocess_command()
