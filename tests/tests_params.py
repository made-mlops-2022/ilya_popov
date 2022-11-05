from dataclasses import dataclass

from ml_project.enities.download_params import DownloadParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.split_params import SplittingParams
from ml_project.enities.training_params import TrainingParams


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


@dataclass()
class TestTrainingPipelineParams:
    input_data_path: str = "tests/train_data_sample.csv"
    output_model_path: str = ""
    metric_path: str = ""
    splitting_params: SplittingParams = SplittingParams(
        test_size=0.3, random_state=40, shuffle=False
    )
    feature_params: FeatureParams = FeatureParams(
        scaler="Standart",
        numerical_features=numerical_features,
        features_to_drop=None,
        target_col="condition",
    )
    train_params: TrainingParams = TrainingParams(random_state=40)
    downloading_params: DownloadParams = None
    use_mlflow: bool = False
    mlflow_uri: str = "Not Implemented"
    mlflow_experiment: str = "Not Implemented"
