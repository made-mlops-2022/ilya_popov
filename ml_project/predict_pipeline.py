import click
import logging
import sys

from ml_project.enities.predict_pipeline_params import read_predict_pipeline_params, PredictPipelineParams
from ml_project.data.make_dataset import make_dataset, predict_to_csv
from ml_project.models.models_fit_predict import deserialize_model, predict_model


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    predict_params = read_predict_pipeline_params(config_path)

    return run_predict_pipeline(predict_params)


def run_predict_pipeline(predict_params: PredictPipelineParams):
    logger.info(f"Start predict pipeline with params: {predict_params}")
    data = make_dataset(predict_params.input_data_path)
    logger.info(f"Data shape = {data.shape}")

    model_pipeline = deserialize_model(predict_params.input_model_path)
    logger.info(f"Deserialized model: {model_pipeline}")

    predicts = predict_model(data, model_pipeline)

    return predict_to_csv(predicts, predict_params.output_data_path)


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
