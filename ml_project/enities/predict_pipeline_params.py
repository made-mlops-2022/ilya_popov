from dataclasses import dataclass

import yaml

from marshmallow_dataclass import class_schema


@dataclass
class PredictPipelineParams:
    input_data_path: str
    input_model_path: str
    output_data_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
