import click

from ml_project.enities.predict_pipeline_params import read_predict_pipeline_params, PredictPipelineParams
from ml_project.data.make_dataset import make_dataset, predict_to_csv
from ml_project.models.models_fit_predict import deserialize_model, predict_model


def predict_pipeline(config_path: str):
    predict_params = read_predict_pipeline_params(config_path)

    return run_predict_pipeline(predict_params)

def run_predict_pipeline(predict_params: PredictPipelineParams):
    data = make_dataset(predict_params.input_data_path)

    model_pipeline = deserialize_model(predict_params.input_model_path)

    predicts = predict_model(data, model_pipeline)

    return predict_to_csv(predicts, predict_params.output_data_path)


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
