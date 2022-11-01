from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split

from ml_project.enities.split_params import SplittingParams


def make_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_train_test_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, test_data = train_test_split(
        data, 
        test_size=params.test_size,
        random_state=params.random_state,
        shuffle=params.shuffle
    )
    return train_data, test_data