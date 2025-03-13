"""
Copyright © 2025 Austin Berrio
Module: tiny.data.downloader
Description: Downloads a file from a given source url to a destinatation path.
"""

import json
import unicodedata
from pathlib import Path

import requests
from tqdm import tqdm


class TinyDownloader:
    def __init__(self, source_url: str, destination_path: str):
        self.url = source_url
        self.path = Path(destination_path)
        self.bar_format = (
            "[{desc}: {percentage:3.0f}%] "
            "[{n_fmt}/{total_fmt}] "
            "[{rate_fmt}{postfix}] "
            "[{elapsed}]"
        )

    def download(self) -> None:
        """Downloads a file from a given URL and saves it locally."""

        response = requests.get(self.url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, bar_format=self.bar_format) as pbar:
            with open(self.path, "wb") as file:
                for data in response.iter_content(1024):
                    pbar.update(len(data))
                    file.write(data)

    def read_text(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        return unicodedata.normalize("NFKC", text)  # Normalize the text

    def read_json(self) -> list[dict[str, any]]:
        with open(self.path, "r", encoding="utf-8") as file:
            return json.load(file)

    def read_parquet(self) -> any:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Install pandas to read Parquet files.")

        return pd.read_parquet(self.path)

    def read(self, data_type: str) -> any:
        if data_type == "text":
            return self.read_text()
        elif data_type == "json":
            return self.read_json()
        elif data_type == "parquet":
            return self.read_parquet()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def read_or_download(self, data_type: str = "text") -> any:
        """Read cached dataset if available, otherwise download it."""
        # Check if the cached path exists
        if self.path.exists():
            print("Reading dataset...")
            return self.read(data_type)

        # Download and cache data
        print("Downloading dataset...")
        self.download()
        return self.read(data_type)
