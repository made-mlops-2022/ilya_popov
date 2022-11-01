import yaml

from dataclasses import dataclass
from typing import Optional
from marshmallow_dataclass import class_schema

from ml_project.enities.download_params import DownloadParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.split_params import SplittingParams
from ml_project.enities.training_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = False
    mlflow_uri: str = "Not Implemented"
    mlflow_experiment: str = "Not Implemented"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))