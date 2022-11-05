import json
import logging
import sys

import click

from ml_project.data.make_dataset import (
    download_files,
    make_dataset,
    split_train_test_data,
)
from ml_project.enities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from ml_project.features.build_features import (
    build_transformer,
    extract_target,
    make_features,
)
from ml_project.models.models_fit_predict import (
    evaluate_metrics,
    make_model_pipeline,
    predict_model,
    serialize_model,
    train_model,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_params = read_training_pipeline_params(config_path)

    return run_train_pipeline(training_params)


def run_train_pipeline(training_params: TrainingPipelineParams):
    logger.info(f"Start train pipeline with params: {training_params}")

    if training_params.downloading_params:
        download_files(training_params.downloading_params, logger)

    data = make_dataset(training_params.input_data_path)
    logger.info(f"Data shape = {data.shape}")
    train_data, test_data = split_train_test_data(
        data, training_params.splitting_params
    )

    logger.info(f"Extracting target: {training_params.feature_params.target_col}")
    y_train = extract_target(train_data, training_params.feature_params)
    y_test = extract_target(test_data, training_params.feature_params)

    x_train = train_data.drop(columns=training_params.feature_params.target_col)
    x_test = test_data.drop(columns=training_params.feature_params.target_col)
    logger.info(f"X train shape = {x_train.shape}, X test shape = {x_test.shape}")

    transformer = build_transformer(training_params.feature_params)
    transformer.fit(x_train)

    train_features = make_features(transformer, x_train)
    logger.info(f"Train features shape = {train_features.shape}")

    model = train_model(train_features, y_train, training_params.train_params)
    model_pipeline = make_model_pipeline(model, transformer)

    predicts = predict_model(x_test, model_pipeline)
    metrics = evaluate_metrics(y_test, predicts)

    with open(training_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metrics: {metrics}")

    path_to_model = serialize_model(model_pipeline, training_params.output_model_path)
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
