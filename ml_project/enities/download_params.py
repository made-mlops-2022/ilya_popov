from dataclasses import dataclass
from typing import List


@dataclass
class DownloadParams:
    service_account_file: str
    scopes: List[str]
    output_folder: str
