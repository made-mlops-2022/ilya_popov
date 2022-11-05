from logging import Logger
from typing import NoReturn, Tuple
from urllib.error import HTTPError

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload

from ml_project.enities.download_params import DownloadParams
from ml_project.enities.split_params import SplittingParams


def download_files(params: DownloadParams, logger: Logger) -> NoReturn:
    try:
        service = make_service(params)
        result = service.files().list(fields="files(id, name)").execute()

        for data_file in result["files"]:
            file_id, filename = data_file.values()

            logger.info(f"Downloading file: {filename}")

            path = f"{params.output_folder}{filename}"
            download_file(path, file_id, service, logger)
    except HTTPError as error:
        logger.error(f"An error is occurred: {error}")


def make_service(params: DownloadParams) -> Resource:
    credentials = service_account.Credentials.from_service_account_file(
        params.service_account_file, scopes=params.scopes
    )
    return build("drive", "v3", credentials=credentials)


def download_file(
    path: str, file_id: str, service: Resource, logger: Logger
) -> NoReturn:
    request = service.files().get_media(fileId=file_id)

    with open(path, "wb") as download_file:
        downloader = MediaIoBaseDownload(download_file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logger.info(f"Download {int(status.progress() * 100)}.")


def make_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_test_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, test_data = train_test_split(
        data,
        test_size=params.test_size,
        random_state=params.random_state,
        shuffle=params.shuffle,
    )
    return train_data, test_data


def predict_to_csv(target: np.ndarray, path: str) -> str:
    data = pd.Series(target)
    data.to_csv(path)
    return path
