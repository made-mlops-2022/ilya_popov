import click

from ml_project.enities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params


def train_pipeline(config_path: str):
    training_params = read_training_pipeline_params(config_path)

    return run_train_pipeline(training_params)

def run_train_pipeline(training_params: TrainingPipelineParams):
    pass


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
