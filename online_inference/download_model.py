import logging

from typing import NoReturn
from urllib.error import HTTPError

from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload

from entities.download_params import DownloadParams


logger = logging.getLogger(__name__)


def download_files(params: DownloadParams) -> NoReturn:
    try:
        service = make_service(params)
        result = service.files().list(fields="files(id, name)").execute()

        for data_file in result["files"]:
            file_id, filename = data_file.values()

            if filename in params.filenames:
                logger.info(f"Downloading file: {filename}")

                path = f"{params.output_folder}{filename}"
                download_file(path, file_id, service)
    except HTTPError as error:
        logger.error(f"An error is occurred: {error}")


def make_service(params: DownloadParams) -> Resource:
    credentials = service_account.Credentials.from_service_account_file(
        params.service_account_file, scopes=params.scopes
    )
    return build("drive", "v3", credentials=credentials)


def download_file(path: str, file_id: str, service: Resource) -> NoReturn:
    request = service.files().get_media(fileId=file_id)

    with open(path, "wb") as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logger.info(f"Download {int(status.progress() * 100)}.")
