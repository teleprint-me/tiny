"""
Copyright © 2025 Austin Berrio
Script: tiny.tinystories
Description: Downloads and converts Tiny Stories to a simple a tiny format.

This script downloads and converts Tiny Stories data into a simplified format suitable for
training a small-scale model. The goal is to obtain around 100 random samples, where each
sample is represented as an input-target pair. The original dataset is in plaintext, containing
short stories, usually 2 - 3 paragraphs in length, and use a simple vocabulary. The plaintext
must be converted to JSON and the dataset must contain sequences within a maximum length.
I'm thinking line by line might be okay per short story. Take each line in pairs. The first line
is the input and the second line is the target. Not sure yet. This is challenging.
"""

import argparse
import json
import os
import random
from pathlib import Path

import requests
from tqdm import tqdm


def get_source_url(dataset_type: str) -> str:
    """Get dataset URL based on type."""
    base_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    return f"{base_url}TinyStories-{dataset_type}.txt?download=true"


def download_dataset(source_url: str, source_path: Path) -> str:
    """Download the dataset from the given URL and save it locally."""
    response = requests.get(source_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(source_path, "wb") as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)

    with open(source_path, "r", encoding="utf-8") as file:
        return file.read()


def read_or_download_dataset(source_url: str, source_path: Path) -> str:
    """Read cached dataset if available, otherwise download it."""
    if source_path.exists():
        print("Reading dataset...")
        with open(source_path, "r", encoding="utf-8") as file:
            return file.read()
    print("Downloading dataset...")
    return download_dataset(source_url, source_path)


def extract_stories(text: str) -> list:
    """
    Extract individual stories from raw text using the <|endoftext|> delimiter.

    Args:
        text (str): Raw dataset text.

    Returns:
        list: List of stories as strings.
    """
    stories = text.split("<|endoftext|>")
    return [story.strip() for story in stories if story.strip()]


def generate_input_target_pairs(story: str, seq_length: int, scale: float) -> list:
    """
    Convert a story into input-target pairs with a tunable sliding window.

    Args:
        story (str): A single story.
        seq_length (int): Number of words per input.
        scale (float): Scale factor controlling step size (0.1 to 2.0).

    Returns:
        list: List of { "input": ..., "target": ... } pairs.
    """
    words = story.split()  # Simple whitespace tokenization
    num_words = len(words)

    pairs = []
    step = max(1, int(seq_length * scale))  # Ensure step is at least 1

    for i in range(0, num_words - seq_length, step):
        input_seq = " ".join(words[i : i + seq_length])
        target_seq = " ".join(words[i + seq_length : i + seq_length + seq_length])

        if target_seq:  # Ensure valid target exists
            pairs.append({"input": input_seq, "target": target_seq})

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Download and preprocess TinyStories dataset.")
    parser.add_argument(
        "--dataset",
        choices=["valid", "train"],
        default="valid",
        help="Choose which dataset to download: 'valid' (default) or 'train'.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to select (default: 100).",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=10,
        help="Length of input sequence in words (default: 10).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor for sliding window (range: 0.1 to 2.0, default: 1.0).",
    )
    parser.add_argument(
        "--output",
        default="data/tinystories.json",
        help="Path to write the output dataset (default: 'data/tinystories.json').",
    )
    return parser.parse_args()


def main():
    """Main function to download, process, and save a subset of the dataset."""
    args = parse_args()

    source_url = get_source_url(args.dataset)
    source_path = Path(f"data/tinystories_{args.dataset}.txt")
    destination_path = Path(args.output)

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    data = read_or_download_dataset(source_url, source_path)

    print("Extracting stories...")
    stories = extract_stories(data)

    print(f"Found {len(stories)} stories. Generating input-target pairs...")
    all_pairs = []

    for story in stories:
        all_pairs.extend(generate_input_target_pairs(story, args.seq_length, args.scale))

    if not all_pairs:
        print("Error: No valid input-target pairs found.")
        return

    if len(all_pairs) < args.samples:
        print(f"Warning: Only {len(all_pairs)} pairs found, using all.")
        sample = all_pairs
    else:
        print(f"Selecting {args.samples} random samples...")
        sample = random.sample(all_pairs, args.samples)

    print(f"Saving dataset to {destination_path}...")
    with open(destination_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print("Done! Dataset saved.")


if __name__ == "__main__":
    main()
