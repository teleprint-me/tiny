"""
Copyright © 2025 Austin Berrio
Script: tiny.hotpot
Description: Downloads and converts the Hotpot QA to a simple a tiny format.

This script downloads and converts Hotpot QA data into a simplified format suitable for
training a small-scale model. The goal is to obtain around 100 random samples, where each
sample is represented as an input-target pair with the question serving as the input and
the answer as the target.
"""

import argparse
import json
import os
import random
from pathlib import Path

import requests
from tqdm import tqdm


def download_dataset(url, source_path):
    """Download the dataset from the given URL and save it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we catch bad responses

    total_size = int(response.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(source_path, "wb") as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)

    # Load and return the JSON after download
    with open(source_path, "r", encoding="utf-8") as file:
        return json.load(file)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Download and preprocess HotpotQA dataset.")
    parser.add_argument(
        "--dataset",
        choices=["dev", "train"],
        default="dev",
        help="Choose which dataset to download: 'dev' (default) or 'train'.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to cherry pick from (Default: 100).",
    )
    parser.add_argument(
        "--output",
        default="data/hotpot.json",
        help="Path to write the output dataset to (Default: 'data/hotpot.json').",
    )
    return parser.parse_args()


def main():
    """Main function to download, process, and save a small subset of the dataset."""

    args = parse_args()

    if args.dataset == "train":
        source_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    else:
        source_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"

    source_path = Path(f"data/hotpot_{args.dataset}.json")

    # Ensure the data directory exists
    destination_path = Path(args.output)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Cache the file to save bandwidth
    if source_path.exists():
        print("Reading dataset...")
        with open(source_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        print("Downloading dataset...")
        data = download_dataset(source_url, source_path)

    print("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(data)

    if not qa_pairs:
        print("Error: No valid QA pairs found in the dataset.")
        return

    if len(qa_pairs) < args.samples:
        print(f"Warning: Only {len(qa_pairs)} valid QA pairs found, using all of them.")
        sample = qa_pairs
    else:
        print(f"Selecting {args.samples} random samples...")
        sample = random.sample(qa_pairs, args.samples)

    print(f"Saving dataset to {destination_path}...")
    with open(destination_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print("Done! Dataset saved.")


if __name__ == "__main__":
    main()
