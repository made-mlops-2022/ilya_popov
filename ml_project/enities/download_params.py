from dataclasses import dataclass


@dataclass
class DownloadParams:
    paths: list[str]
    output_folder: str
    s3_bucket: str