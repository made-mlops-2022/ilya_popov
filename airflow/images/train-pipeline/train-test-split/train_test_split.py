import os

from typing import Tuple

import click
import pandas as pd

from sklearn.model_selection import train_test_split


def split_train_test_data(
    data: pd.DataFrame, test_size: float, random_state: int, shuffle: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return train_data, test_data


@click.command()
@click.argument("input_dir")
@click.option(
    "-t",
    "--test_size",
    default=0.2,
    type=float,
    help="Proportion of the dataset to include in the test split",
)
@click.option("-r", "--random_state", default=777, type=int)
@click.option(
    "-s", "--shuffle", default=False, type=bool, help="Shuffle data before splitting"
)
def split_command(input_dir: str, test_size: float, random_state: int, shuffle: bool):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    train_data, test_data = split_train_test_data(
        data, test_size, random_state, shuffle
    )

    train_data.to_csv(os.path.join(input_dir, "train_data.csv"))
    test_data.to_csv(os.path.join(input_dir, "test_data.csv"))


if __name__ == "__main__":
    split_command()
