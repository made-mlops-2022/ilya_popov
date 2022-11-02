import json
import click

from ml_project.data.make_dataset import make_dataset, split_train_test_data
from ml_project.enities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from ml_project.features.build_features import build_transformer, extract_target, make_features
from ml_project.models.models_fit_predict import evaluate_metrics, make_model_pipeline, predict_model, serialize_model, train_model


def train_pipeline(config_path: str):
    training_params = read_training_pipeline_params(config_path)

    return run_train_pipeline(training_params)

def run_train_pipeline(training_params: TrainingPipelineParams):
    data = make_dataset(training_params.input_data_path)
    train_data, test_data = split_train_test_data(data, training_params.splitting_params)

    y_train = extract_target(train_data, training_params.feature_params)
    y_test = extract_target(test_data, training_params.feature_params)
    # print(train_data.columns)
    x_train = train_data.drop(columns=training_params.feature_params.target_col)
    x_test = test_data.drop(columns=training_params.feature_params.target_col)

    transformer = build_transformer(training_params.feature_params)
    transformer.fit(x_train)

    train_features = make_features(transformer, x_train)

    model = train_model(train_features, y_train, training_params.train_params)
    model_pipeline = make_model_pipeline(model, transformer)

    predicts = predict_model(x_test, model_pipeline)
    metrics = evaluate_metrics(y_test, predicts)

    with open(training_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    path_to_model = serialize_model(model_pipeline, training_params.output_model_path)
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
