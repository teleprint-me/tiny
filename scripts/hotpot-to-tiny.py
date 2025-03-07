"""
Script: hotpot-to-tiny.py
Description: Downloads and converts the Hotpot QA to a simple a tiny format.

---

We only need a few hundred samples at most. Any more than that is overkill.
Ideally, we just grab 100 samples random. The output file is formatted for
input-target pairs where the question becomes the input and the answer becomes
the target.
"""

import json
import os
import random
from pathlib import Path

import requests
from tqdm import tqdm

# Constants
SOURCE_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
SOURCE_PATH = Path("data/hotpot.json")
DESTINATION_PATH = Path("data/tiny.json")
SAMPLE_SIZE = 100

# Ensure the data directory exists
os.makedirs(os.path.dirname(DESTINATION_PATH), exist_ok=True)


def download_dataset(url):
    """Download the dataset from the given URL and return the JSON data."""
    response = requests.get(url)

    total_size = int(response.headers.get("content-length", 0)) / (32 * 1024)
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(SOURCE_PATH, "wb") as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)

    response.raise_for_status()  # Raise an error if the download fails

    return response.json()


def extract_qa_pairs(data):
    """Extracts question-answer pairs from the dataset."""
    qa_pairs = []
    for entry in data:
        try:
            question = entry["question"].strip()
            answer = entry["answer"].strip()
            if question and answer:  # Ensure non-empty strings
                qa_pairs.append({"input": question, "target": answer})
        except KeyError:
            continue  # Skip malformed entries
    return qa_pairs


def main():
    """Main function to download, process, and save a small subset of the dataset."""

    # Cache the file to save bandwidth
    if SOURCE_PATH.exists():
        print("Reading dataset...")
        with open(SOURCE_PATH, "r") as file:
            data = json.load(file)
    else:
        print("Downloading dataset...")
        data = download_dataset(SOURCE_URL)

    print("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(data)

    if len(qa_pairs) < SAMPLE_SIZE:
        print(f"Warning: Only {len(qa_pairs)} valid QA pairs found, using all of them.")
        sample = qa_pairs
    else:
        print(f"Selecting {SAMPLE_SIZE} random samples...")
        sample = random.sample(qa_pairs, SAMPLE_SIZE)

    print(f"Saving dataset to {DESTINATION_PATH}...")
    with open(DESTINATION_PATH, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print("Done! Dataset saved.")


if __name__ == "__main__":
    main()
