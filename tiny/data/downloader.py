"""
Copyright © 2025 Austin Berrio
Module: tiny.data.downloader
Description: Downloads a file from a given source url to a destinatation path.

This is intentionally kept as simple as possible. The downloader times out after 30
seconds (which is the default) and it's allowed to raise an exception on failure. The
failure is reported to the user and the user is expected to manually intervene.

The TinyDataDownloader:
  - checks if the path exists to mitigate the need to list every possible exception.
  - creates a directory and its parent structure if it does not exist.
  - only supports plaintext, JSON, and parquet to keep things simple.
    - text is not restricted to `.txt` files and includes any valid unicode text file.
  - does not make any assumptions for the cached or returned data.
  - simplifies how progress is reported while downloading data.
This keeps the code lean and clean as a result.
"""

import json
import os
import unicodedata

import pandas as pd
import requests
from tqdm import tqdm

from tiny.logger import TinyLogger


class TinyDataDownloader:
    def __init__(self, root_dir: str = "data", verbose: bool = False):
        os.makedirs(os.path.dirname(root_dir), exist_ok=True)
        self.dir = root_dir
        self.encoding = "utf-8"
        self.logger = TinyLogger.get_logger(self.__class__.__name__, verbose)
        self.bar_format = (
            "[{desc}: {percentage:3.0f}%] "
            "[{n_fmt}/{total_fmt}] "
            "[{rate_fmt}{postfix}] "
            "[{elapsed}]"
        )

    def download(self, source_url: str, source_file: str) -> None:
        """Downloads a file from a given URL and saves it locally."""

        self.logger.info(f"Downloading '{source_file}' from '{source_url}'.")

        response = requests.get(source_url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, bar_format=self.bar_format) as pbar:
            with open(source_file, "wb") as file:
                for data in response.iter_content(1024):
                    pbar.update(len(data))
                    file.write(data)

    def read_text(self, source_file: str) -> str:
        """Read and normalize a plaintext file to unicode utf-8."""

        with open(source_file, "r", encoding=self.encoding) as file:
            text = file.read()
        return unicodedata.normalize("NFKC", text)  # Normalize the text

    def read_json(self, source_file: str) -> any:
        """Read and normalize a JSON file to unicode utf-8."""

        with open(source_file, "r", encoding=self.encoding) as file:
            return json.load(file)

    def read_parquet(self, source_file: str) -> any:
        """Read a parquet file."""

        return pd.read_parquet(source_file)

    def read_file(self, source_file: str, file_type: str) -> any:
        """Read a supported file type into memory."""

        self.logger.info(f"Reading '{file_type}' for '{source_file}'.")

        if file_type == "text":
            return self.read_text(source_file)
        elif file_type == "json":
            return self.read_json(source_file)
        elif file_type == "parquet":
            return self.read_parquet(source_file)

        raise ValueError(f"Unsupported file type: {file_type}")

    def read_or_download(self, source_url: str, source_file: str, file_type: str = "text") -> any:
        """Read cached file type if available, otherwise download it."""
        # Check if the cached path exists
        if os.path.exists(source_file):
            return self.read_file(source_file, file_type)

        # Otherwise cache the path
        self.download(source_url, source_file)
        return self.read_file(source_file, file_type)
