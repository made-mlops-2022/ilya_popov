import io
import os

import click
import pandas as pd

from faker import Faker


def generate_data(num_rows: int = 10) -> pd.DataFrame:
    fake = Faker()

    fake.set_arguments("age", {"min_value": 18, "max_value": 100})
    fake.set_arguments("sex", {"min_value": 0, "max_value": 1})
    fake.set_arguments("cp", {"min_value": 0, "max_value": 3})
    fake.set_arguments("trestbps", {"min_value": 90, "max_value": 200})
    fake.set_arguments("chol", {"min_value": 120, "max_value": 600})
    fake.set_arguments("fbs", {"min_value": 0, "max_value": 1})
    fake.set_arguments("restecg", {"min_value": 0, "max_value": 2})
    fake.set_arguments("thalach", {"min_value": 70, "max_value": 200})
    fake.set_arguments("exang", {"min_value": 0, "max_value": 1})
    fake.set_arguments("oldpeak", {"min_value": 0, "max_value": 6})
    fake.set_arguments("slope", {"min_value": 0, "max_value": 2})
    fake.set_arguments("ca", {"min_value": 0, "max_value": 3})
    fake.set_arguments("thal", {"min_value": 0, "max_value": 2})
    fake.set_arguments("condition", {"min_value": 0, "max_value": 1})

    fake_data = pd.read_csv(
        io.StringIO(
            fake.csv(
                header=[
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
                    "condition",
                ],
                data_columns=(
                    "{{pyint:age}}",
                    "{{pyint:sex}}",
                    "{{pyint:cp}}",
                    "{{pyint:trestbps}}",
                    "{{pyint:chol}}",
                    "{{pyint:fbs}}",
                    "{{pyint:restecg}}",
                    "{{pyint:thalach}}",
                    "{{pyint:exang}}",
                    "{{pyfloat:oldpeak}}",
                    "{{pyint:slope}}",
                    "{{pyint:ca}}",
                    "{{pyint:thal}}",
                    "{{pyint:condition}}",
                ),
                num_rows=num_rows,
            )
        )
    )
    return fake_data


@click.command()
@click.argument("output_dir")
@click.option("-c", "--count", default=10, help="Number of rows in the dataset")
def generate_command(output_dir: str, count: int = 10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o777)

    data = generate_data(count)
    data.to_csv(os.path.join(output_dir, "data.csv"))


if __name__ == "__main__":
    generate_command()
