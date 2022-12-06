from dataclasses import dataclass


@dataclass
class DownloadParams:
    service_account_file: str
    scopes: list[str]
    filenames: list[str]
    output_folder: str
