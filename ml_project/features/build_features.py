import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml_project.enities.feature_params import FeatureParams


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(params.scaler),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def build_numerical_pipeline(scaler_type: str) -> Pipeline:
    match scaler_type:
        case "MinMax":
            scaler = MinMaxScaler()
        case "Standart":
            scaler = StandardScaler()

    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scale", scaler),
        ]
    )
    return num_pipeline


def extract_target(data: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return data[params.target_col]
