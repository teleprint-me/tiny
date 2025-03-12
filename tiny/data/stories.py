"""
Copyright © 2025 Austin Berrio
Script: tiny.data.stories
Description: Downloads and converts Tiny Stories to a simple tiny format.

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
import re
import unicodedata
from pathlib import Path

import requests
from tqdm import tqdm


def tqdm_bar_format() -> str:
    """Customizes the progress indicator and removes the bar for a cleaner output."""
    return "[{desc}: {percentage:3.0f}%] [{n_fmt}/{total_fmt}] [{rate_fmt}{postfix}] [{elapsed}]"


def get_source_url(dataset_type: str) -> str:
    base_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    return f"{base_url}TinyStories-{dataset_type}.txt?download=true"


def download_dataset(source_url: str, source_path: Path) -> str:
    """Downloads the dataset from the given URL and saves it locally."""
    response = requests.get(source_url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    bar_format = tqdm_bar_format()
    with tqdm(total=total, unit="B", unit_scale=True, bar_format=bar_format) as progress_bar:
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
            text = file.read()
    else:
        print("Downloading dataset...")
        text = download_dataset(source_url, source_path)
    return unicodedata.normalize("NFKC", text)  # Normalize Unicode


def extract_stories(text: str) -> list[str]:
    """Extract individual stories from raw text using <|endoftext|> as delimiter."""
    stories = text.split("<|endoftext|>")
    return [story.strip() for story in stories if story.strip()]


def preprocess_story_lines(story: str) -> list[str]:
    """
    Process a story into a list of **clean, individual sentences**.

    - Handles quotes properly.
    - Ensures punctuation splits are correct.
    - Removes any artifacts from bad splitting.

    Returns:
        list[str]: A properly segmented list of sentences.
    """
    sentences = []
    current_sentence = []
    open_quote = False  # Track if a quote is open

    for line in story.splitlines():
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        tokens = re.split(r'(["“”])', line)  # Split by quote characters while keeping them
        for token in tokens:
            if token in ('"', "“", "”"):
                open_quote = not open_quote  # Toggle quote state
                current_sentence.append(token)
            elif open_quote:
                current_sentence.append(token)  # Inside quote, don't break
            else:
                parts = re.split(r"([.!?])", token)  # Split by sentence-ending punctuation
                for part in parts:
                    if part in ".!?":
                        current_sentence.append(part)
                        sentences.append("".join(current_sentence).strip())  # Fix space issue
                        current_sentence = []  # Reset for new sentence
                    elif part:
                        current_sentence.append(part)

    if current_sentence:
        sentences.append("".join(current_sentence).strip())  # Store remaining sentence

    return sentences


def preprocess_stories(stories: list[str]) -> list[list[str]]:
    """Convert each story into a clean list of sentences."""
    return [preprocess_story_lines(story) for story in stories]


def clean_ascii(text: str) -> str:
    """
    Converts curly quotes and apostrophes to standard ASCII equivalents.
    This ensures clean, consistent formatting across all environments.
    """
    replacements = {
        "’": "'",  # Fancy apostrophe → ASCII apostrophe
        "‘": "'",  # Single open quote → ASCII apostrophe
        "”": '"',  # Fancy close double quote → ASCII double quote
        "“": '"',  # Fancy open double quote → ASCII double quote
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_sentence_pairs(
    story_sentences: list[str], input_size: int = 2, target_size: int = 1
) -> list:
    """
    Convert preprocessed sentences into structured input-target pairs.

    - Explicitly joins **exactly** `input_size` sentences in input.
    - Ensures **exactly** `target_size` sentences in target.
    - Cleans text to **avoid unicode inconsistencies**.

    Args:
        story_sentences (list[str]): A preprocessed list of sentences.
        input_size (int): Number of sentences per input.
        target_size (int): Number of sentences per target.

    Returns:
        list: List of {"input": ..., "target": ...} pairs.
    """

    skipped = 0
    pairs = []
    for i in range(0, len(story_sentences), input_size + target_size):
        input_sentences = " ".join(story_sentences[i : i + input_size]).strip()
        target_sentences = " ".join(
            story_sentences[i + input_size : i + input_size + target_size]
        ).strip()

        if 3 > len(input_sentences) or 3 > len(target_sentences):
            skipped += 1
            continue

        # Ensure strict enforcement of sizes and clean ASCII
        input_sentences = clean_ascii(input_sentences)
        target_sentences = clean_ascii(target_sentences)

        if input_sentences and target_sentences:
            pairs.append({"input": input_sentences, "target": target_sentences})

    print(f"Skipped {skipped} malformed input-target pair")
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
        "--all-pairs",
        action="store_true",
        help="Select all samples. Overrides --samples when True (Default: False).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=2,
        help="Joins up to `input_size` sentences in input (Default: 2).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1,
        help="Joins up to `target_size` sentences in target (Default: 1).",
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

    print(f"Processing {len(stories)} stories...")
    processed_stories = preprocess_stories(stories)

    print("Generating input-target pairs...")
    all_pairs = []
    for story_sentences in processed_stories:
        sentence_pairs = generate_sentence_pairs(story_sentences, args.input_size, args.target_size)
        all_pairs.extend(sentence_pairs)
    print(f"Generated {len(all_pairs)} samples.")

    if args.all_pairs or len(all_pairs) < args.samples:
        print(f"Warning: Using all {len(all_pairs)} pairs found.")
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
