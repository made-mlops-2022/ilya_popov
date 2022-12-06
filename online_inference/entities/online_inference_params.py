from typing import Optional
import yaml

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from entities.download_params import DownloadParams


@dataclass
class OnlineInferenceParams:
    download_params: Optional[DownloadParams]
    model_path: str = "models/model.pkl"


OnlineInferenceParamsSchema = class_schema(OnlineInferenceParams)


def read_online_inference_params(path: str) -> OnlineInferenceParams:
    with open(path, "r") as input_stream:
        schema = OnlineInferenceParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
